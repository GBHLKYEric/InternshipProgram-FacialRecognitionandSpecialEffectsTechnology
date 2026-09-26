"""Plot all 40 CelebA annotation frequencies from the verified full-corpus report.

These are dataset-provided labels, including subjective labels; no claim of
objective traits or population representativeness is made.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=Path('reports/celeba-expanded-data.json'))
    parser.add_argument('--output', type=Path, default=Path('reports/celeba-distribution'))
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding='utf-8'))
    total = report['full_corpus_image_count']
    counts = report['full_corpus_positive_attribute_counts']
    if total <= 0 or len(counts) != 40 or any(not 0 <= value <= total for value in counts.values()):
        raise ValueError('Expected 40 valid annotation counts from the full dataset')
    rows = sorted(counts.items(), key=lambda row: row[1])
    selected = {'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'}
    fig, ax = plt.subplots(figsize=(10, 12), layout='constrained')
    values = [100*count/total for _, count in rows]
    ax.barh([name for name, _ in rows], values, color=['#2b785a' if name in selected else '#acbbae' for name, _ in rows])
    for index, value in enumerate(values):
        ax.text(value+.5, index, f'{value:.1f}%', va='center', fontsize=8)
    ax.set(xlim=(0, 100), xlabel='Positive annotation frequency (%)',
           title=f'CelebA: {total:,} images / 40 labels\nGreen: five StarGAN targets; labels are not mutually exclusive')
    ax.spines[['top', 'right']].set_visible(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix('.svg'))
    fig.savefig(args.output.with_suffix('.png'), dpi=140)
    plt.close(fig)
    result = {'source': str(args.report), 'images': total, 'labels': len(counts),
              'selected_attributes': {key: {'positive': counts[key], 'fraction': counts[key]/total} for key in sorted(selected)},
              'interpretation': 'Dataset annotation frequencies; labels overlap and some are subjective. Not estimates of a general population.'}
    args.output.with_suffix('.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
