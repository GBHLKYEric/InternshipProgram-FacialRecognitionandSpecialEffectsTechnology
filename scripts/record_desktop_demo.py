"""Record actual native-window operations using only the public sample image.

No camera is opened. Video is a fixed-5-FPS UI walkthrough, not a speed benchmark.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont, ImageGrab
import tkinter as tk
from desktop import Desktop, EFFECT_LABELS
from app import STARGAN_MODEL


def main():
    root = tk.Tk()
    app = Desktop(root, testing=True)
    output = ROOT.parent/'本地桌面程序演示.mp4'
    writer = imageio_ffmpeg.write_frames(str(output), (1200, 900), fps=5, codec='libx264',
        pix_fmt_out='yuv420p', macro_block_size=1, quality=8,
        output_params=['-threads', '1', '-movflags', '+faststart'])
    writer.send(None)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 22)
    state = {'frames': 0, 'scene': -1, 'started': time.perf_counter(), 'scene_started': None,
             'title': '', 'events': [], 'error': None, 'video_started': None, 'last_frame': None,
             'gl_pending': False, 'gl_capture_count': 0, 'gl_image': None,
             'completed_at': None, 'finished': False, 'scene_events_start': 0}
    gl_frame = ROOT/'runs/desktop-video-review/current-opengl.png'
    gl_frame.parent.mkdir(parents=True, exist_ok=True)

    def effect(label):
        app.last_frame_metrics = None
        app.effect.set(label); app.apply_settings()

    def generate():
        if not STARGAN_MODEL.is_file():
            raise FileNotFoundError('Video requires the locally exported StarGAN checkpoint')
        app.last_result = None
        app.gan_button.invoke()

    def completed():
        index = state['scene']
        if index < 6:
            return app.last_frame_metrics is not None and app.last_frame_metrics['effect'] == list(EFFECT_LABELS.values())[index]
        if index == 6:
            return app.last_result is not None and 'cosine' in app.last_result
        if index == 7:
            return app.last_result is not None and 'model_ms' in app.last_result
        if index == 8:
            return app.mesh_ready
        if index == 9:
            return any(e['event'] == 'captured' and abs(e['degrees']) > 5
                       for e in app.mesh_events[state['scene_events_start']:])
        return app.mesh_process is None and app.worker.released

    scenes = [(label+'：公开示例逐项处理', lambda label=label: effect(label), 2.5)
              for label in EFFECT_LABELS]
    scenes += [
        ('双图验证：同图比较仅检查数值一致性', lambda: app.send('verify', (app.latest.copy(), app.latest.copy(), .363)), 3),
        ('StarGAN：左侧输入裁剪，右侧目标属性合成', generate, 5),
        ('3DDFA：当前公开图像的三维重建', lambda: app.mesh_button.invoke(), 5),
        ('原生三维窗口：旋转查看统计估计的人脸', lambda: None, 5),
        ('完成：摄像头功能已另作真实设备测试', lambda: (app.close_mesh(), app.sample_button.invoke()), 3),
    ]

    def finish():
        if state['finished']:
            return
        state['finished'] = True
        try:
            writer.close()
        except Exception as exc:
            state['error'] = f'Video encoder close failed: {exc}'
        finally:
            app.close()
        capture = cv2.VideoCapture(str(output))
        decoded = 0
        while capture.read()[0]:
            decoded += 1
        capture.release()
        if decoded != state['frames']:
            state['error'] = f'decoded {decoded}, expected {state["frames"]}'
        report = {'output': output.name, 'codec': 'H.264', 'pixel_format': 'yuv420p',
                  'width': 1200, 'height': 900, 'playback_fps': 5, 'frames': decoded,
                  'playback_seconds': decoded/5, 'capture_wall_seconds': time.perf_counter()-state['started'],
                  'camera_opened': False, 'input': 'public NASA sample only',
                  'scope': 'Actual Tk window captures and explicitly requested native OpenGL framebuffer captures. 5 FPS time grid repeats the previous frame if capture is late; not algorithm throughput evidence.',
                  'events': state['events'], 'error': state['error'], 'camera_released': app.worker.released,
                  'all_scenes_acknowledged': len(state['events']) == len(scenes) and all('completed_frame' in e for e in state['events'])}
        if state['error'] is None and not report['all_scenes_acknowledged']:
            state['error'] = report['error'] = 'Video ended before every scene acknowledged completion'
        (ROOT/'reports/desktop-video.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False), flush=True)
        state['exit_code'] = 1 if state['error'] else 0

    def tick():
        try:
            if app.last_error:
                raise RuntimeError(app.last_error)
            if not app.ready or app.latest is None:
                root.after(200, tick); return
            now = time.perf_counter()
            if state['video_started'] is None:
                state['video_started'] = now
            if state['scene'] >= 0 and state['completed_at'] is None:
                if completed():
                    state['completed_at'] = now
                    state['events'][-1]['completed_frame'] = state['frames']
                elif now-state['scene_started'] > 45:
                    raise TimeoutError('Scene did not complete: '+state['title'])
            if state['scene'] < 0 or (state['completed_at'] is not None and now-state['completed_at'] >= scenes[state['scene']][2]):
                state['scene'] += 1
                if state['scene'] == len(scenes):
                    finish(); return
                title, action, _ = scenes[state['scene']]
                state['title'], state['scene_started'] = title, now
                state['completed_at'] = None
                state['scene_events_start'] = len(app.mesh_events)
                action()
                state['events'].append({'title': title, 'frame': state['frames']})
            captures = sum(e['event'] == 'captured' for e in app.mesh_events)
            if captures > state['gl_capture_count']:
                # Read only after the child's acknowledgement, before sending
                # the next request that can overwrite this temporary file.
                with Image.open(gl_frame) as buffer:
                    state['gl_image'] = buffer.convert('RGB')
                state['gl_capture_count'] = captures
                state['gl_pending'] = False
            if state['scene'] in (8, 9) and app.mesh_ready and not state['gl_pending']:
                if state['scene'] == 9:
                    app.mesh_pipe.send({'command': 'yaw', 'degrees': -45+90*min(1, (now-state['scene_started'])/5)})
                app.mesh_pipe.send({'command': 'capture', 'path': str(gl_frame)})
                state['gl_pending'] = True
            if state['scene'] in (8, 9) and state['gl_image'] is not None:
                shot = state['gl_image'].copy()
            else:
                root.update_idletasks()
                shot = ImageGrab.grab(window=int(root.wm_frame(), 16)).convert('RGB')
            shot.thumbnail((1180, 816))
            frame = Image.new('RGB', (1200, 900), '#e8eee8')
            draw = ImageDraw.Draw(frame)
            draw.text((22, 12), state['title'], fill='#214633', font=font)
            frame.paste(shot, ((1200-shot.width)//2, 50+(816-shot.height)//2))
            draw.text((22, 869), '本地原生窗口 · 公开示例 · 5 FPS 演示编码，不代表算法处理帧率', fill='#385546', font=font)
            expected_frame = int((now-state['video_started'])*5)
            while state['frames'] < expected_frame and state['last_frame'] is not None:
                writer.send(state['last_frame'])
                state['frames'] += 1
            state['last_frame'] = frame.tobytes()
            writer.send(state['last_frame'])
            state['frames'] += 1
            root.after(200, tick)
        except Exception as exc:
            state['error'] = f'{type(exc).__name__}: {exc}'
            finish()
    root.after(200, tick)
    def close_recording():
        state['error'] = 'Recording window closed before completion'
        finish()
    root.protocol('WM_DELETE_WINDOW', close_recording)
    root.mainloop()
    raise SystemExit(state.get('exit_code', 1))


if __name__ == '__main__':
    main()
