"""Transfer COCO RetinaNet to the full WIDER train split in the MMDet container.

Default: one full epoch, a frozen COCO-pretrained ResNet50 backbone, trainable
FPN/detection towers and newly initialized one-class prediction layer. This is
a reproducible CPU transfer baseline, not a claim of a converged face model.
It also exports predictions for the separate official WIDER difficulty metric.
"""
import argparse
import hashlib
from itertools import islice
import json
from pathlib import Path
import time
import urllib.request

import torch
from mmengine.config import Config
from mmengine.hooks import Hook
from mmengine.registry import HOOKS, LOOPS
from mmengine.runner import Runner
from mmengine.runner.loops import EpochBasedTrainLoop
from mmdet.datasets.samplers import AspectRatioBatchSampler
from mmdet.registry import DATA_SAMPLERS
from mmdet.utils import register_all_modules

WEIGHT_NAME = 'retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
WEIGHT_SHA = 'c2398f9ec0843ed9a29e72d7a788741abbf2b4250e64f8f793092eaecaa0ab4f'
WEIGHT_URL = ('https://download.openmmlab.com/mmdetection/v2.0/retinanet/'
              'retinanet_r50_fpn_1x_coco/'+WEIGHT_NAME)


@DATA_SAMPLERS.register_module()
class WiderResumeBatchSampler(AspectRatioBatchSampler):
    """Preserve original deterministic batches, skipping before image decoding.

    Keep the original length so epoch/learning-rate/checkpoint counters retain
    their meaning. Only the iterator used for a resumed partial epoch is shorter.
    """
    skip_batches = 0

    def __iter__(self):
        self._aspect_ratio_buckets = [[], []]
        return islice(super().__iter__(), self.skip_batches, None)


@LOOPS.register_module()
class WiderResumeEpochLoop(EpochBasedTrainLoop):
    """MMEngine's default epoch loop replays a prefix after an iteration resume.

    This loop preserves the restored global iteration and resumes at the next
    batch. It does not promise bitwise-identical random augmentation after a
    process restart; weights, optimizer and learning-rate state are restored.
    """
    def run_epoch(self):
        batches_per_epoch = len(self.dataloader)
        offset = self._iter-self._epoch*batches_per_epoch
        if not 0 <= offset <= batches_per_epoch:
            raise ValueError('Checkpoint iteration is incompatible with this data/batch configuration')
        sampler = self.dataloader.batch_sampler
        if not isinstance(sampler, WiderResumeBatchSampler):
            raise TypeError('Resume loop requires WiderResumeBatchSampler')
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()
        sampler.skip_batches = offset
        try:
            for idx, data_batch in enumerate(self.dataloader, start=offset):
                self.run_iter(idx, data_batch)
        finally:
            sampler.skip_batches = 0
        self.runner.call_hook('after_train_epoch')
        self._epoch += 1


def check_resume():
    """Framework-level check that a middle-epoch resume decodes no prefix images."""
    from types import SimpleNamespace
    from mmengine.dataset import DefaultSampler
    from torch.utils.data import DataLoader, Dataset

    class TinyDataset(Dataset):
        loaded = []
        def __len__(self):
            return 12
        def get_data_info(self, index):
            return {'width': 1 if index % 3 else 2, 'height': 2}
        def __getitem__(self, index):
            self.loaded.append(index)
            return index

    dataset = TinyDataset()
    sampler = DefaultSampler(dataset, shuffle=True, seed=42)
    batch_sampler = WiderResumeBatchSampler(sampler, batch_size=2)
    all_batches = list(batch_sampler)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=0)
    loop = WiderResumeEpochLoop.__new__(WiderResumeEpochLoop)
    loop._runner = SimpleNamespace(model=torch.nn.Identity(), call_hook=lambda *a, **k: None)
    loop.dataloader, loop._iter, loop._epoch = loader, 2, 0
    batch_indices = []
    def record_iteration(batch_idx, data_batch):
        batch_indices.append(batch_idx)
        loop._iter += 1
    loop.run_iter = record_iteration
    loop.run_epoch()
    expected = [index for batch in all_batches[2:] for index in batch]
    assert dataset.loaded == expected, (dataset.loaded, expected)
    assert batch_indices == list(range(2, len(loader)))
    assert loop._iter == len(loader) and loop._epoch == 1
    result = {'status': 'passed', 'original_batches': all_batches, 'resumed_from_iteration': 2,
              'loaded_indices': dataset.loaded, 'expected_indices': expected,
              'already_completed_images_reloaded': False, 'final_iteration': loop._iter,
              'scope': 'real MMEngine sampler + PyTorch DataLoader; no model quality claim'}
    Path('reports').mkdir(exist_ok=True)
    Path('reports/wider-resume-check.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


@HOOKS.register_module()
class WiderProgressAndPredictions(Hook):
    def before_train(self, runner):
        self.start = time.perf_counter()
        self.start_iter = runner.iter
        self.prediction_dir = Path(runner.work_dir)/'wider_predictions'
        self.prediction_dir.mkdir(parents=True, exist_ok=True)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        for name, value in (outputs or {}).items():
            if 'loss' in name and not torch.isfinite(torch.as_tensor(value)).all():
                raise FloatingPointError(f'Non-finite {name} at iteration {runner.iter+1}; stop before treating this as progress')
        if (runner.iter+1) % 25:
            return
        elapsed = time.perf_counter()-self.start
        completed = runner.iter+1-self.start_iter
        report = {'status': 'training', 'iteration': runner.iter+1, 'total_iterations': runner.max_iters,
                  'elapsed_seconds_current_run': elapsed,
                  'seconds_per_iteration_current_run': elapsed/completed,
                  'estimated_remaining_seconds': (runner.max_iters-runner.iter-1)*elapsed/completed,
                  'threads': torch.get_num_threads(), 'work_dir': runner.work_dir}
        Path('reports/wider-mmdet-full-progress.json').write_text(json.dumps(report, indent=2))

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        for output in outputs:
            original = Path(output.img_path)
            prediction = output.pred_instances.cpu()
            boxes = prediction.bboxes.numpy().copy()
            boxes[:, 2:4] -= boxes[:, :2]
            scores = prediction.scores.numpy()
            path = self.prediction_dir/original.parent.name/(original.stem+'.txt')
            path.parent.mkdir(parents=True, exist_ok=True)
            lines = [original.stem, str(len(scores))]
            lines.extend(' '.join(f'{float(v):.8f}' for v in [*box, score]) for box, score in zip(boxes, scores))
            path.write_text('\n'.join(lines)+'\n')


def main(args):
    torch.set_num_threads(args.threads)
    register_all_modules(init_default_scope=True)
    Path('reports').mkdir(exist_ok=True)
    work = Path(args.output)
    work.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, expected in [('train', 12880), ('val', 3226)]:
        payload = json.loads(Path(f'data/wider/full_{split}.json').read_text())
        counts[split] = {'images': len(payload['images']), 'boxes': len(payload['annotations'])}
        if counts[split]['images'] != expected:
            raise ValueError(f'Refusing incomplete {split} split')
    weight = Path('data/wider-full')/WEIGHT_NAME
    if not weight.exists():
        weight.parent.mkdir(parents=True, exist_ok=True)
        partial = weight.with_suffix('.pth.incomplete')
        with urllib.request.urlopen(WEIGHT_URL, timeout=60) as response, partial.open('wb') as target:
            while chunk := response.read(1024*1024):
                target.write(chunk)
        if hashlib.sha256(partial.read_bytes()).hexdigest() != WEIGHT_SHA:
            raise ValueError('Downloaded pretrained detector SHA256 mismatch; partial file preserved')
        partial.replace(weight)
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    if digest != WEIGHT_SHA:
        raise ValueError('Official pretrained detector SHA256 mismatch')
    original = torch.load(weight, map_location='cpu', weights_only=True)
    state = original['state_dict']
    discarded = ['bbox_head.retina_cls.weight', 'bbox_head.retina_cls.bias']
    for key in discarded:
        if key not in state:
            raise ValueError('Unexpected source checkpoint structure')
    transfer = {key: value for key, value in state.items() if key not in discarded}
    adapted = work/'coco-transfer-initialization.pth'
    torch.save({'state_dict': transfer}, adapted)
    cfg = Config.fromfile('research/configs/wider_retinanet.py')
    cfg.model.backbone.init_cfg = None
    cfg.model.backbone.frozen_stages = 4
    cfg.model.backbone.norm_eval = True
    cfg.model.test_cfg.score_thr = .001
    cfg.model.test_cfg.max_per_img = 1000
    cfg.train_dataloader.batch_size = args.batch_size
    cfg.train_dataloader.batch_sampler = dict(type='WiderResumeBatchSampler')
    for split in ['train', 'val']:
        loader = cfg[f'{split}_dataloader']
        loader.num_workers = 0
        loader.persistent_workers = False
        loader.dataset.ann_file = f'wider/full_{split}.json'
    cfg.train_dataloader.dataset.filter_cfg = dict(filter_empty_gt=False, min_size=1)
    cfg.train_dataloader.dataset.pipeline = [
        dict(type='LoadImageFromFile'), dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', scale=(args.size, args.size), keep_ratio=True),
        dict(type='RandomFlip', prob=.5), dict(type='PackDetInputs')]
    cfg.val_dataloader.dataset.pipeline = [
        dict(type='LoadImageFromFile'), dict(type='Resize', scale=(args.size, args.size), keep_ratio=True),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))]
    cfg.test_dataloader = cfg.val_dataloader
    cfg.val_evaluator.ann_file = 'data/wider/full_val.json'
    cfg.test_evaluator = cfg.val_evaluator
    cfg.train_cfg.max_epochs = args.epochs
    cfg.train_cfg.type = 'WiderResumeEpochLoop'
    # The final epoch normally triggers validation regardless of val_interval.
    # Defer it here; the single explicit runner.val() below records its metrics.
    cfg.train_cfg.val_begin = args.epochs+1
    cfg.train_cfg.val_interval = args.epochs+1
    cfg.param_scheduler = [dict(type='LinearLR', start_factor=.01, by_epoch=False, begin=0, end=250)]
    cfg.optim_wrapper.optimizer.lr = args.lr
    cfg.optim_wrapper.clip_grad = dict(max_norm=10., norm_type=2, error_if_nonfinite=True)
    cfg.auto_scale_lr = dict(enable=False, base_batch_size=args.batch_size)
    cfg.env_cfg.dist_cfg.backend = 'gloo'
    cfg.randomness = dict(seed=42, deterministic=False)
    cfg.default_hooks.logger.interval = 25
    cfg.default_hooks.checkpoint = dict(type='CheckpointHook', by_epoch=False,
        interval=100, max_keep_ckpts=2, save_last=True)
    cfg.custom_hooks = [dict(type='WiderProgressAndPredictions')]
    cfg.work_dir = str(work)
    cfg.load_from = args.resume or str(adapted)
    cfg.resume = bool(args.resume)
    if args.resume:
        # Only accept a local checkpoint from this experiment configuration.
        # MMEngine restores its optimizer/scheduler; CLI LR changes alone do not
        # redefine that restored state. Threads may change; batch order may not.
        checkpoint = torch.load(args.resume, map_location='cpu')
        config_text = checkpoint['meta'].get('cfg') or checkpoint['meta'].get('config')
        if not config_text:
            raise ValueError('Checkpoint has no saved configuration for the resume contract')
        previous = Config.fromstring(config_text, file_format='.py')
        for name in ['batch_size', 'dataset.ann_file']:
            old, new = previous.train_dataloader, cfg.train_dataloader
            for component in name.split('.'):
                old, new = old[component], new[component]
            if old != new:
                raise ValueError(f'Resume cannot change train_dataloader.{name}')
        old_pipeline = previous.train_dataloader.dataset.pipeline
        if old_pipeline != cfg.train_dataloader.dataset.pipeline:
            raise ValueError('Resume cannot change training preprocessing')
        if previous.randomness.seed != cfg.randomness.seed:
            raise ValueError('Resume cannot change the shuffle seed')
        if previous.optim_wrapper.optimizer.lr != cfg.optim_wrapper.optimizer.lr:
            raise ValueError('Resume must retain the original optimizer learning rate')
        del checkpoint
    cfg.dump(str(work/'effective-config.py'))
    record = {'status': 'prepared', 'model': 'RetinaNet ResNet50 FPN', 'dataset': counts,
              'training': vars(args), 'pretrained_url': WEIGHT_URL, 'pretrained_sha256': digest,
              'initialization': 'COCO backbone/FPN/towers/regression loaded; one-class retina_cls newly initialized',
              'frozen': 'entire ResNet50 backbone; FPN and heads remain trainable',
              'stability': '250-step linear learning-rate warmup, norm-10 gradient clipping, reject non-finite gradients/loss',
              'resume_policy': 'skip previously completed deterministic aspect-ratio batches; restore optimizer/scheduler; random augmentation after process restart is not guaranteed bitwise identical',
              'discarded_source_parameters': discarded, 'metric': 'Full validation COCO AP, separate WIDER difficulty evaluation',
              'limitations': 'One CPU transfer epoch at limited resolution does not establish convergence.'}
    report_path = Path('reports/wider-mmdet-full.json')
    report_path.write_text(json.dumps(record, indent=2))
    runner = Runner.from_cfg(cfg)
    record['trainable_parameters'] = sum(p.numel() for p in runner.model.parameters() if p.requires_grad)
    start = time.perf_counter()
    runner.train()
    record['training_seconds'] = time.perf_counter()-start
    validation_start = time.perf_counter()
    record['coco_metrics'] = runner.val()
    record['validation_seconds'] = time.perf_counter()-validation_start
    record['status'] = 'completed'
    record['prediction_files'] = sum(1 for _ in (work/'wider_predictions').glob('*/*.txt'))
    if record['prediction_files'] != 3226:
        raise ValueError('Full validation prediction export incomplete')
    report_path.write_text(json.dumps(record, indent=2, default=float))
    Path('reports/wider-mmdet-full-progress.json').write_text(json.dumps({'status': 'completed', 'report': str(report_path)}))
    print(json.dumps(record, indent=2, default=float), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--size', type=int, default=320)
    parser.add_argument('--lr', type=float, default=.0005)
    parser.add_argument('--output', default='runs/wider-mmdet-full')
    parser.add_argument('--resume')
    parser.add_argument('--check-resume', action='store_true')
    parsed = parser.parse_args()
    check_resume() if parsed.check_resume else main(parsed)
