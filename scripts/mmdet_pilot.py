"""Train and evaluate a small real WIDER subset in the separate MMDet container.

This deliberately short CPU experiment validates the training/evaluation path.
It is neither a converged detector nor the official WIDER easy/medium/hard AP.
"""
import json
from pathlib import Path
import time

import torch
import mmcv
import mmdet
import mmengine
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.utils import register_all_modules


def main():
    root = Path.cwd()
    for split in ['train', 'val']:
        if not (root / f'data/wider/{split}.json').is_file():
            raise FileNotFoundError('First run python -m research.fetch_wider_subset')
    torch.set_num_threads(4)
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile('research/configs/wider_retinanet.py')
    cfg.model.backbone.init_cfg = None  # no unreported ImageNet pretrained weights
    # Keep ranked low-confidence candidates so the evaluator can report actual AP.
    # A scratch model's ~0.01 scores otherwise all disappear at the default .05.
    cfg.model.test_cfg.score_thr = 0.0
    cfg.train_dataloader.batch_size = 2
    cfg.train_dataloader.num_workers = 0
    cfg.train_dataloader.persistent_workers = False
    cfg.val_dataloader.num_workers = 0
    cfg.val_dataloader.persistent_workers = False
    cfg.test_dataloader = cfg.val_dataloader
    cfg.train_dataloader.dataset.pipeline = [
        dict(type='LoadImageFromFile'), dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', scale=(320, 320), keep_ratio=True), dict(type='RandomFlip', prob=.5),
        dict(type='PackDetInputs')]
    cfg.val_dataloader.dataset.pipeline = [
        dict(type='LoadImageFromFile'), dict(type='Resize', scale=(320, 320), keep_ratio=True),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='PackDetInputs', meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor'))]
    cfg.train_cfg.max_epochs = 1
    cfg.train_cfg.val_interval = 2  # evaluate explicitly once after the one training epoch
    cfg.param_scheduler = []
    cfg.auto_scale_lr = dict(enable=False, base_batch_size=2)
    cfg.env_cfg.dist_cfg.backend = 'gloo'
    cfg.randomness = dict(seed=42, deterministic=False)
    cfg.default_hooks.logger.interval = 1
    cfg.default_hooks.checkpoint.interval = 1
    cfg.work_dir = 'runs/wider_pilot'
    start = time.perf_counter()
    runner = Runner.from_cfg(cfg)
    runner.train()
    metrics = runner.val()
    from mmdet.apis import inference_detector
    import cv2
    runner.model.cfg = cfg
    validation = json.loads((root/'data/wider/val.json').read_text())
    sample_path = root/'data/WIDER_val/images'/validation['images'][0]['file_name']
    prediction = inference_detector(runner.model, str(sample_path)).pred_instances.cpu()
    canvas = cv2.imread(str(sample_path))
    candidates = prediction.scores.argsort(descending=True)[:10]
    for index in candidates:
        x1, y1, x2, y2 = prediction.bboxes[index].numpy().astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 180, 255), 2)
        cv2.putText(canvas, f'{float(prediction.scores[index]):.3f}', (x1, max(15, y1)),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 180, 255), 1)
    Path('reports').mkdir(exist_ok=True)
    cv2.imwrite('reports/wider_pilot_prediction.jpg', canvas)
    counts = {split: len(json.loads((root/f'data/wider/{split}.json').read_text())['images'])
              for split in ['train', 'val']}
    report = {'purpose': 'CPU training/inference/evaluation pipeline pilot on real WIDER subset',
              'dataset_images': counts, 'epochs': 1, 'seed': 42, 'input_resize': [320, 320],
              'initialization': 'random, no pretrained backbone', 'model': 'RetinaNet ResNet50 FPN, one face class',
              'evaluation_score_threshold': 0.0,
              'wall_seconds': time.perf_counter()-start, 'metrics': metrics,
              'metric_protocol': 'COCO AP 0.50:0.95 on selected validation images; NOT official WIDER AP',
              'versions': {'torch': torch.__version__, 'mmcv': mmcv.__version__,
                           'mmengine': mmengine.__version__, 'mmdet': mmdet.__version__},
              'inference_visualization': 'Top 10 candidate boxes, including low scores, from un-converged pilot model',
              'limitations': '32/16 images and one epoch do not establish generalization or convergence.'}
    Path('reports').mkdir(exist_ok=True)
    Path('reports/wider_pilot.json').write_text(json.dumps(report, indent=2, default=float), encoding='utf-8')
    print(json.dumps(report, indent=2, default=float))


if __name__ == '__main__':
    main()
