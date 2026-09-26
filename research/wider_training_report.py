"""Export actual MMEngine training scalars to CSV and a scientific curve.

Loss values are the recorded up-to-50-step log-window averages, not per-image
loss. Learning rate is the value reported at that logging iteration.
This exporter supports a running snapshot and never labels a partial run complete.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def run(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sources = sorted(Path(args.work).glob('*/vis_data/scalars.json'))
    if not sources:
        raise FileNotFoundError('No actual MMEngine scalar files found')
    rows = {}
    hashes = {}
    for source in sources:
        content = source.read_bytes()
        hashes[str(source)] = hashlib.sha256(content).hexdigest()
        lines = content.splitlines()
        for index, line in enumerate(lines):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines)-1 and not content.endswith(b'\n'):
                    continue  # Snapshot during an unfinished final write.
                raise
            if 'loss' not in record:
                continue
            if not all(math.isfinite(record[name]) for name in ['loss', 'loss_cls', 'loss_bbox']):
                raise ValueError('Non-finite loss found; do not plot a successful training curve')
            rows[record['iter']] = record
    ordered = [rows[index] for index in sorted(rows)]
    if not ordered:
        raise ValueError('No actual training loss values found')
    prefix = Path(args.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    columns = ['iter', 'epoch', 'loss', 'loss_cls', 'loss_bbox', 'lr', 'grad_norm', 'time', 'data_time']
    with prefix.with_suffix('.csv').open('w', encoding='utf-8', newline='') as destination:
        writer = csv.DictWriter(destination, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(ordered)
    experiment = json.loads(Path('reports/wider-mmdet-full.json').read_text())
    complete = experiment.get('status') == 'completed'
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, layout='constrained',
                             gridspec_kw={'height_ratios': [3, 1]})
    steps = [row['iter'] for row in ordered]
    for key, label in [('loss', 'Total'), ('loss_cls', 'Classification'), ('loss_bbox', 'Box regression')]:
        axes[0].plot(steps, [row[key] for row in ordered], label=label)
    state = 'completed full training + validation' if complete else 'in-progress snapshot'
    axes[0].set(title=f'WIDER RetinaNet transfer: {state}', ylabel='Loss (up-to-50-step window mean)')
    axes[0].legend()
    axes[0].grid(alpha=.2)
    axes[1].plot(steps, [row['lr'] for row in ordered], color='#8e44ad')
    axes[1].set(xlabel='Completed training iteration', ylabel='Learning rate')
    axes[1].grid(alpha=.2)
    fig.savefig(prefix.with_suffix('.svg'))
    plt.close(fig)
    report = {'status': 'completed' if complete else 'in_progress_snapshot',
              'logged_points': len(ordered), 'first_iteration': steps[0], 'last_iteration': steps[-1],
              'first_logged_loss': ordered[0]['loss'], 'last_logged_loss': ordered[-1]['loss'],
              'scalar_file_sha256': hashes,
              'negative_logged_data_time_points': sum(row.get('data_time', 0) < 0 for row in ordered),
              'limitations': 'Loss uses recorded up-to-50-step log-window averages, generally emitted every 25 steps; learning rate is the reported value at that iteration. Decreasing train loss does not establish validation quality or convergence. Negative wall-clock data_time is preserved, not used for throughput.'}
    if complete:
        saved = (Path(args.work)/'last_checkpoint').read_text().strip()
        local = Path(saved.removeprefix('/project/'))
        checkpoint_hash = hashlib.sha256()
        with local.open('rb') as checkpoint:
            while chunk := checkpoint.read(1024*1024):
                checkpoint_hash.update(chunk)
        report['final_checkpoint'] = str(local)
        report['final_checkpoint_sha256'] = checkpoint_hash.hexdigest()
        report['final_checkpoint_bytes'] = local.stat().st_size
    prefix.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', default='runs/wider-mmdet-full-stable')
    parser.add_argument('--output', default='reports/wider-mmdet-training-curves')
    run(parser.parse_args())
