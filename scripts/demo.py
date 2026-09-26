"""Reproducible animated input + real per-frame inference, not a webcam recording."""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import Vision


def main():
    output = ROOT / 'reports'
    output.mkdir(exist_ok=True)
    sample = cv2.imread(str(ROOT / 'assets/sample.jpg'))
    if sample is None:
        raise FileNotFoundError('Run scripts/fetch_models.py first')
    vision = Vision()
    effects = ['none', 'glasses', 'crown', 'beauty', 'lipstick', 'all']
    fps, segment = 20, 60
    path = output / 'demo.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1024, 620))
    if not writer.isOpened():
        raise RuntimeError('MP4 writer unavailable')
    measurements, detected, samples = [], 0, []
    try:
        for index in range(segment * len(effects)):
            effect = effects[index // segment]
            transform = cv2.getRotationMatrix2D((256, 256), 9*np.sin(index/35), 1+.025*np.sin(index/24))
            transform[:, 2] += [12*np.sin(index/20), 5*np.sin(index/30)]
            frame = cv2.warpAffine(sample, transform, (512, 512), borderMode=cv2.BORDER_REFLECT)
            result, metrics = vision.process(frame, effect, .7, effect == 'none')
            measurements.append(metrics['pipeline_ms'])
            detected += metrics['faces'] > 0
            canvas = np.full((620, 1024, 3), (31, 42, 37), np.uint8)
            canvas[55:567, :512], canvas[55:567, 512:] = frame, result
            cv2.putText(canvas, 'FACE VISION LAB | CPU | '+effect.upper(), (28, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (224, 242, 220), 2)
            cv2.putText(canvas, 'Input: animated NASA public sample', (20, 590), cv2.FONT_HERSHEY_SIMPLEX, .55, (215, 225, 210), 1)
            cv2.putText(canvas, f'Real inference: {metrics["pipeline_ms"]:.1f} ms | faces {metrics["faces"]}', (534, 590), cv2.FONT_HERSHEY_SIMPLEX, .55, (215, 225, 210), 1)
            writer.write(canvas)
            if index % segment == segment // 2:
                image_path = output / f'effect-{effect}.jpg'
                cv2.imwrite(str(image_path), canvas)
                samples.append(str(image_path.name))
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    readable, _ = capture.read()
    capture.release()
    if frame_count != segment*len(effects) or not readable:
        raise RuntimeError('Video readback failed')
    result = {'video':path.name, 'source':'NASA astronaut public-domain sample; affine animated input, not a live camera',
              'frames':frame_count, 'playback_fps':fps, 'duration_seconds':frame_count/fps,
              'frames_with_face':detected, 'effects':effects, 'samples':samples,
              'pipeline_ms_median':float(np.median(measurements)), 'pipeline_ms_p95':float(np.percentile(measurements,95)),
              'pipeline_mean_fps':float(1000/np.mean(measurements)),
              'timing_scope':'YuNet detection + effects only; excludes decode/encode/browser display',
              'recorded_at':time.strftime('%Y-%m-%dT%H:%M:%S')}
    (output / 'demo-metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
