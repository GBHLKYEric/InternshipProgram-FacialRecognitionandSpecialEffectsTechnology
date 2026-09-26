"""Controlled CPU thread comparison; updates are discarded, never main training.

Run only while the real training container is paused. Every candidate starts
from the same checkpoint, optimizer state, and already-loaded real image batches.
It measures train_step compute, not full-epoch quality or file-loading speed.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.utils import register_all_modules

# Registers resume-aware batching without invoking its main training entrypoint.
import mmdet_full  # noqa: F401


def main(args):
    torch.set_num_interop_threads(1)
    torch.set_num_threads(4)
    register_all_modules(init_default_scope=True)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config_text = checkpoint['meta'].get('cfg') or checkpoint['meta'].get('config')
    if not config_text:
        raise ValueError('Checkpoint has no saved configuration')
    cfg = Config.fromstring(config_text, file_format='.py')
    cfg.work_dir = 'runs/wider-thread-benchmark/discarded-work'
    cfg.custom_hooks = []
    cfg.train_dataloader.num_workers = 0
    cfg.train_dataloader.persistent_workers = False
    cfg.train_dataloader.batch_sampler = dict(type='WiderResumeBatchSampler')
    runner = Runner.from_cfg(cfg)
    loader = runner.train_dataloader
    start_batch = checkpoint['meta']['iter'] % len(loader)
    loader.batch_sampler.skip_batches = start_batch
    iterator = iter(loader)
    batches = [next(iterator) for _ in range(args.batches)]
    results = []
    for threads in args.threads:
        torch.set_num_threads(threads)
        runner.model.load_state_dict(checkpoint['state_dict'], strict=True)
        runner.model.train()
        optimizer = runner.build_optim_wrapper(cfg.optim_wrapper)
        optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
        samples, losses = [], []
        for index in range(args.warmup+args.repeats):
            data = copy.deepcopy(batches[index % len(batches)])
            start = time.perf_counter()
            output = runner.model.train_step(data, optimizer)
            seconds = time.perf_counter()-start
            loss = float(output['loss'])
            if not torch.isfinite(torch.tensor(loss)):
                raise FloatingPointError('Non-finite disposable benchmark loss')
            if index >= args.warmup:
                samples.append(seconds)
                losses.append(loss)
        results.append({'threads': threads, 'seconds': samples, 'losses': losses,
                        'median_seconds': statistics.median(samples),
                        'mean_seconds': statistics.mean(samples),
                        'min_seconds': min(samples), 'max_seconds': max(samples)})
        print(json.dumps(results[-1]), flush=True)
        del optimizer
    base = results[0]['median_seconds']
    for result in results:
        result['median_speedup_vs_first_candidate'] = base/result['median_seconds']
    report = {'status': 'completed', 'checkpoint': str(checkpoint_path),
              'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
              'checkpoint_iteration': checkpoint['meta']['iter'], 'start_batch': start_batch,
              'sample_batches': args.batches, 'batch_size': cfg.train_dataloader.batch_size,
              'face_counts_by_batch': [[len(sample.gt_instances) for sample in batch['data_samples']]
                                       for batch in batches],
              'image_ids_by_batch': [[sample.img_id for sample in batch['data_samples']]
                                      for batch in batches],
              'warmup_steps_per_candidate': args.warmup, 'measured_steps_per_candidate': args.repeats,
              'results': results,
              'limitations': 'Short local compute-only comparison on a fixed set of real batches. Main process paused externally. No checkpoint updates from this benchmark are retained; not a full-epoch throughput guarantee.'}
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--threads', type=int, nargs='+', default=[4, 6, 8])
    parser.add_argument('--batches', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=16)
    parser.add_argument('--output', default='reports/wider-thread-benchmark.json')
    main(parser.parse_args())
