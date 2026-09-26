"""Run a complete WIDER validation baseline and its official difficulty protocol.

The protocol follows the author-distributed eval_tools/evaluation.m:
IoU >= .5, inclusive pixel arithmetic, global score normalization, 1000
thresholds, difficulty-specific ignored faces and precision-envelope AP.
This is a transparent Python implementation, not a claim of a MATLAB run.
Reference: http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/eval_script/eval_tools.zip
OpenCV's independent Python reference: opencv_zoo/tools/eval/datasets/widerface.py
No fabricated detections are inserted for images with no detections.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.io import loadmat

from research.wider_download import filesystem_path, sha256


def load_ground_truth(root):
    directory = filesystem_path(Path(root)/'eval_tools/ground_truth')
    main = loadmat(directory/'wider_face_val.mat')
    difficulty = {name: loadmat(directory/f'wider_{name}_val.mat')['gt_list']
                  for name in ['easy', 'medium', 'hard']}
    items = []
    for event_i, event in enumerate(main['event_list']):
        event_name = str(event[0][0])
        for image_i, image in enumerate(main['file_list'][event_i][0]):
            image_name = str(image[0][0])
            boxes = main['face_bbx_list'][event_i][0][image_i][0].astype(np.float64).reshape(-1, 4)
            keep = {name: masks[event_i][0][image_i][0].astype(np.int64).ravel()-1
                    for name, masks in difficulty.items()}
            if any(np.any((indices < 0) | (indices >= len(boxes))) for indices in keep.values()):
                raise ValueError('Invalid difficulty ground-truth indices')
            items.append((event_name, image_name, boxes, keep))
    if len(items) != 3226:
        raise ValueError(f'Refusing partial/different validation protocol: {len(items)} images')
    return items, {p.name: sha256(p) for p in directory.glob('wider*_val.mat')}


def write_prediction(path, image_name, boxes):
    path = filesystem_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [image_name, str(len(boxes))]
    lines.extend(' '.join(f'{float(x):.8f}' for x in row) for row in boxes)
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')


def read_prediction(path):
    lines = filesystem_path(path).read_text(encoding='utf-8').splitlines()
    if len(lines) < 2 or len(lines)-2 != int(lines[1]):
        raise ValueError(f'Malformed prediction count: {path}')
    if lines[0].strip() not in [Path(path).stem, Path(path).stem+'.jpg']:
        raise ValueError(f'Prediction header does not match its image name: {path}')
    boxes = np.asarray([[float(x) for x in line.split()] for line in lines[2:]], dtype=np.float64).reshape(-1, 5)
    if not np.isfinite(boxes).all() or np.any(boxes[:, 2:4] < 0):
        raise ValueError('Non-finite prediction or negative extent')
    return boxes[np.argsort(-boxes[:, 4], kind='stable')]


def predict_yunet(args, items):
    cv2.setNumThreads(args.threads)
    detector = cv2.FaceDetectorYN.create(str(Path(args.model).resolve()), '', (320, 320),
        score_threshold=args.score, nms_threshold=args.nms, top_k=args.top_k)
    output = Path(args.prediction_dir)
    config = {'model_sha256': sha256(args.model), 'score': args.score, 'nms': args.nms,
              'top_k': args.top_k, 'input': 'original resolution, single scale, no flips',
              'opencv': cv2.__version__, 'threads': args.threads, 'images_expected': len(items)}
    output.mkdir(parents=True, exist_ok=True)
    manifest = output/'configuration.json'
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError('Existing prediction directory uses a different configuration')
    manifest.write_text(json.dumps(config, indent=2), encoding='utf-8')
    elapsed = []
    start = time.perf_counter()
    for index, (event, name, _, _) in enumerate(items):
        destination = output/event/(name+'.txt')
        if args.resume and filesystem_path(destination).exists():
            read_prediction(destination)
            continue
        path = filesystem_path(Path(args.root)/'WIDER_val/images'/event/(name+'.jpg'))
        image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Image could not be decoded: {path}')
        detector.setInputSize((image.shape[1], image.shape[0]))
        t = time.perf_counter()
        faces = detector.detect(image)[1]
        elapsed.append(time.perf_counter()-t)
        boxes = np.empty((0, 5)) if faces is None else np.column_stack((faces[:, :4], faces[:, -1]))
        boxes = boxes[np.argsort(-boxes[:, 4], kind='stable')]
        write_prediction(destination, name, boxes)
        if (index+1) % 100 == 0 or index+1 == len(items):
            print(f'YuNet {index+1}/{len(items)} images, wall {time.perf_counter()-start:.1f}s', flush=True)
    return {**config, 'new_images_this_run': len(elapsed), 'wall_seconds_this_run': time.perf_counter()-start,
            'inference_median_ms_this_run': float(np.median(elapsed)*1000) if elapsed else None,
            'inference_p95_ms_this_run': float(np.percentile(elapsed, 95)*1000) if elapsed else None}


def box_iou_inclusive(boxes, truth):
    """WIDER MATLAB convention: xywh -> xyxy with x2=x+w, then +1 areas."""
    first = boxes[:, :4].copy()
    second = truth.copy()
    first[:, 2:4] += first[:, :2]
    second[:, 2:4] += second[:, :2]
    lower = np.maximum(first[:, None, :2], second[None, :, :2])
    upper = np.minimum(first[:, None, 2:4], second[None, :, 2:4])
    intersection = np.maximum(upper-lower+1, 0).prod(axis=2)
    area_a = (first[:, 2:4]-first[:, :2]+1).prod(axis=1)
    area_b = (second[:, 2:4]-second[:, :2]+1).prod(axis=1)
    return intersection/(area_a[:, None]+area_b[None, :]-intersection)


def image_counts(prediction, truth, keep, thresholds=1000):
    counts = np.zeros((thresholds, 2), dtype=np.float64)
    if not len(prediction) or not len(truth):
        return counts  # matches official handling of empty GT images
    valid = np.zeros(len(truth), dtype=bool)
    valid[keep] = True
    overlap = box_iou_inclusive(prediction, truth)
    best = overlap.argmax(axis=1)
    matched = overlap[np.arange(len(prediction)), best] >= .5
    proposals = np.ones(len(prediction), dtype=np.int64)
    true_positives = np.zeros(len(prediction), dtype=np.int64)
    recalled = set()
    for p in range(len(prediction)):
        if matched[p]:
            g = int(best[p])
            if not valid[g]:
                proposals[p] = 0
            elif g not in recalled:
                recalled.add(g)
                true_positives[p] = 1
    proposal_sum, recall_sum = proposals.cumsum(), true_positives.cumsum()
    values = 1-np.arange(1, thresholds+1)/thresholds
    ends = np.searchsorted(-prediction[:, 4], -values, side='right')-1
    exists = ends >= 0
    counts[exists, 0] = proposal_sum[ends[exists]]
    counts[exists, 1] = recall_sum[ends[exists]]
    return counts


def average_precision(recall, precision):
    r = np.r_[0., recall, 1.]
    p = np.r_[0., precision, 0.]
    p = np.maximum.accumulate(p[::-1])[::-1]
    changed = np.flatnonzero(r[1:] != r[:-1])
    return float(np.sum((r[changed+1]-r[changed])*p[changed+1]))


def evaluate(items, directory):
    predictions = {}
    for event, name, _, _ in items:
        predictions[event, name] = read_prediction(Path(directory)/event/(name+'.txt'))
    nonempty = [x for x in predictions.values() if len(x)]
    if nonempty:
        low = min(float(x[:, 4].min()) for x in nonempty)
        high = max(float(x[:, 4].max()) for x in nonempty)
        for p in nonempty:
            p[:, 4] = (p[:, 4]-low)/(high-low) if high > low else 1.
    metrics, curves = {}, {}
    for difficulty in ['easy', 'medium', 'hard']:
        totals = np.zeros((1000, 2))
        faces = 0
        for event, name, truth, keep in items:
            faces += len(keep[difficulty])
            totals += image_counts(predictions[event, name], truth, keep[difficulty])
        precision = np.divide(totals[:, 1], totals[:, 0], out=np.zeros(1000), where=totals[:, 0] > 0)
        recall = totals[:, 1]/faces
        metrics[difficulty] = {'ap': average_precision(recall, precision), 'ground_truth_faces': faces,
            'recall_at_lowest_included_score': float(recall[-1]),
            'precision_at_lowest_included_score': float(precision[-1])}
        curves[difficulty] = {'precision': precision.tolist(), 'recall': recall.tolist()}
        print(difficulty, metrics[difficulty], flush=True)
    return metrics, curves, {'images': len(predictions), 'images_without_detections': len(predictions)-len(nonempty),
                             'total_detections': sum(len(x) for x in predictions.values())}


def selfcheck():
    truth = np.array([[0., 0., 10., 10.], [20., 20., 10., 10.]])
    pred = np.array([[20., 20., 10., 10., 1.], [0., 0., 10., 10., .9], [0., 0., 10., 10., .8]])
    assert np.allclose(np.diag(box_iou_inclusive(np.c_[truth, [1., 1.]], truth)), 1)
    # ignored hit contributes neither proposal nor recall; duplicate becomes FP.
    assert np.array_equal(image_counts(pred, truth, np.array([0]))[-1], [2, 1])
    assert average_precision(np.array([0., 1.]), np.array([1., 1.])) == 1.


def plot_curves(report_path):
    """Render measured precision/recall data, never substitute preset curves."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding='utf-8'))
    curves = json.loads(report_path.with_suffix('.curves.json').read_text(encoding='utf-8'))
    fig, axis = plt.subplots(figsize=(7, 5), layout='constrained')
    for difficulty in ['easy', 'medium', 'hard']:
        values = curves[difficulty]
        ap = report['metrics'][difficulty]['ap']
        axis.plot(values['recall'], values['precision'], label=f'{difficulty}: AP {ap:.4f}')
    axis.set(xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1.02),
             title='Full WIDER validation: 3,226 images, official difficulty masks')
    axis.legend(loc='lower left')
    axis.grid(alpha=.2)
    fig.savefig(report_path.with_name(report_path.stem+'-pr.svg'))
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='data')
    parser.add_argument('--prediction-dir', default='runs/wider-yunet-full/predictions')
    parser.add_argument('--output', default='reports/wider-yunet-full.json')
    parser.add_argument('--model', default='models/face_detection_yunet_2023mar.onnx')
    parser.add_argument('--predict-yunet', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--score', type=float, default=.3)
    parser.add_argument('--nms', type=float, default=.45)
    parser.add_argument('--top-k', type=int, default=5000)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    selfcheck()
    items, hashes = load_ground_truth(args.root)
    inference = predict_yunet(args, items) if args.predict_yunet else None
    start = time.perf_counter()
    metrics, curves, counts = evaluate(items, args.prediction_dir)
    report = {'protocol': 'WIDER official validation difficulty masks; Python port of author MATLAB protocol',
              'iou': .5, 'thresholds': 1000, 'full_validation': True, 'counts': counts,
              'ground_truth_sha256': hashes, 'metrics': metrics, 'inference': inference,
              'evaluation_seconds': time.perf_counter()-start,
              'limitations': 'Pretrained baseline or supplied checkpoint; not proof of self-trained convergence. No official leaderboard submission.'}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    output.with_suffix('.curves.json').write_text(json.dumps(curves), encoding='utf-8')
    plot_curves(output)
    print(json.dumps(report, indent=2), flush=True)
