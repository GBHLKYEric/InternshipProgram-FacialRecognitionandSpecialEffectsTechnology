"""Encode the complete measured demo as H.264/yuv420p for common video players."""
import json
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT/'reports/demo.mp4'
    target = ROOT.parent/'项目演示视频.mp4'
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-i', str(source),
                    '-an', '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', str(target)], check=True)
    metrics = json.loads((ROOT/'reports/demo-metrics.json').read_text(encoding='utf-8'))
    capture = cv2.VideoCapture(str(target))
    count = 0
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    while capture.read()[0]:
        count += 1
    capture.release()
    if count != metrics['frames'] or abs(fps-metrics['playback_fps']) > .01:
        raise RuntimeError('Export changed the number of frames or playback speed.')
    result = {'file': target.name, 'codec': 'H.264', 'pixel_format': 'yuv420p',
              'frames_decoded': count, 'fps': fps, 'width': width, 'height': height,
              'seconds': count/fps, 'ffmpeg': imageio_ffmpeg.get_ffmpeg_version(),
              'scope': 'Re-encoding only. Model timing remains the original measured run.'}
    (ROOT/'reports/video-export.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
