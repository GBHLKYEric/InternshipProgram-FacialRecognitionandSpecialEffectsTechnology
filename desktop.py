"""Native Tk desktop application. Camera frames stay in memory; no HTTP server.

Run: python desktop.py. Run actual GUI/camera checks: python desktop.py --self-test
"""
from __future__ import annotations

import argparse
import json
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from app import ROOT, STARGAN_MODEL, Vision, decode_image

EFFECT_LABELS = {'原图与检测': 'none', '眼镜': 'glasses', '皇冠': 'crown',
                 '美颜': 'beauty', '口红': 'lipstick', '组合特效': 'all'}


def read_image(path):
    """Read Unicode Windows paths without relying on OpenCV's filename handling."""
    source = Path(path)
    if source.stat().st_size > 8_000_000:
        raise ValueError('图片不得超过 8 MB')
    frame = cv2.imdecode(np.frombuffer(source.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if frame is None or max(frame.shape[:2]) > 4096:
        raise ValueError('请选择边长不超过 4096 的 JPEG 或 PNG 图片')
    scale = min(1., 960 / max(frame.shape[:2]))
    return cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame


class Worker(threading.Thread):
    """One owner for mutable Vision models and the camera; Tk stays on main thread."""
    def __init__(self):
        super().__init__(daemon=True)
        self.commands = queue.Queue()
        self.events = queue.Queue()
        self.frames = queue.Queue(maxsize=1)
        self.closing = threading.Event()
        self.camera = None
        self.token = 0
        self.settings = ('all', .5, True)
        self.source = None
        self.released = True

    def stop_camera(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.released = True

    def emit_frame(self, raw, started, camera=False):
        result, metrics = self.vision.process(raw, *self.settings)
        metrics['effect'], metrics['strength'], metrics['show_landmarks'] = self.settings
        item = (self.token, raw, result, metrics, started, camera)
        try:
            self.frames.get_nowait()
        except queue.Empty:
            pass
        self.frames.put_nowait(item)

    def run(self):
        try:
            self.vision = Vision()
            self.events.put(('ready', 0, None))
            while not self.closing.is_set():
                try:
                    command, token, value = self.commands.get(timeout=.005 if self.camera else .05)
                except queue.Empty:
                    command = None
                if command:
                    self.token = token
                    try:
                        if command == 'settings':
                            self.settings = value
                            if self.source is not None and self.camera is None:
                                self.emit_frame(self.source, time.perf_counter())
                        elif command == 'camera':
                            self.stop_camera()
                            for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
                                capture = cv2.VideoCapture(value, backend)
                                if capture.isOpened():
                                    self.camera = capture
                                    break
                                capture.release()
                            if self.camera is None:
                                raise ValueError('无法打开摄像头。请检查设备序号、Windows 相机权限以及其他程序是否占用。')
                            self.released = False
                            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            self.camera.set(cv2.CAP_PROP_FPS, 30)
                            self.events.put(('camera', token, {'backend': self.camera.getBackendName(), 'index': value}))
                        elif command == 'stop':
                            self.stop_camera()
                            self.events.put(('stopped', token, None))
                        elif command == 'image':
                            self.stop_camera()
                            self.source = value
                            self.emit_frame(self.source, time.perf_counter())
                            self.events.put(('image', token, None))
                        elif command in ('verify', 'gan', 'mesh'):
                            self.stop_camera()
                            if command == 'verify':
                                result = self.vision.verify(*value)
                            elif command == 'gan':
                                result = self.vision.edit_attributes(*value)
                            else:
                                from vision3d.reconstruct import reconstruct
                                faces = self.vision.detect(value)
                                if len(faces) != 1:
                                    raise ValueError('三维重建需要恰好一张清晰人脸')
                                result = reconstruct(value, faces[0, :4])
                            self.events.put((command, token, result))
                    except Exception as exc:
                        self.stop_camera()
                        self.events.put(('error', token, f'{type(exc).__name__}: {exc}'))
                if self.camera is not None:
                    try:
                        start = time.perf_counter()
                        ok, frame = self.camera.read()
                        if not ok:
                            raise RuntimeError('摄像头读取失败，设备已释放。')
                        self.source = cv2.flip(frame, 1)
                        self.emit_frame(self.source, start, camera=True)
                    except Exception as exc:
                        self.stop_camera()
                        self.events.put(('error', self.token, f'{type(exc).__name__}: {exc}'))
        except Exception as exc:
            self.events.put(('fatal', self.token, f'{type(exc).__name__}: {exc}'))
        finally:
            self.stop_camera()


class Desktop:
    def __init__(self, root, testing=False):
        self.root, self.testing = root, testing
        self.token = 0
        self.ready = self.camera_active = False
        self.unavailable = False
        self.latest = None
        self.last_result = None
        self.last_frame_metrics = None
        self.received = []
        self.camera_samples = []
        self.camera_info = {}
        self.benchmark_start = None
        self.last_error = None
        self.mesh_result = None
        self.mesh_process = self.mesh_pipe = None
        self.mesh_events = []
        self.mesh_ready = False
        self.worker = Worker()
        root.title('智能视觉 AI 实验室 · 本地桌面程序')
        root.geometry('1160x800')
        root.minsize(1000, 730)
        root.configure(bg='#f4f6f2')
        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('.', font=('Microsoft YaHei UI', 10))
        style.configure('TFrame', background='#f4f6f2')
        style.configure('TLabel', background='#f4f6f2', foreground='#213c31')
        style.configure('TButton', padding=(10, 7))
        header = ttk.Frame(root, padding=(22, 14)); header.pack(fill='x')
        ttk.Label(header, text='智能视觉 AI 实验室', font=('Microsoft YaHei UI', 22, 'bold')).pack(anchor='w')
        ttk.Label(header, text='本机摄像头与图片 · 检测 / 特效 / 双图验证 / 属性编辑 / 三维重建').pack(anchor='w', pady=(3, 0))
        controls = ttk.Frame(root, padding=(22, 0, 22, 10)); controls.pack(fill='x')
        self.camera_index = tk.StringVar(value='0')
        ttk.Label(controls, text='摄像头').pack(side='left')
        ttk.Combobox(controls, width=3, state='readonly', textvariable=self.camera_index, values=('0', '1', '2', '3')).pack(side='left', padx=6)
        self.start_button = ttk.Button(controls, text='开启摄像头', command=self.start_camera)
        self.start_button.pack(side='left', padx=3)
        self.stop_button = ttk.Button(controls, text='停止摄像头', command=self.stop_camera)
        self.stop_button.pack(side='left', padx=3)
        ttk.Button(controls, text='打开图片', command=self.open_image).pack(side='left', padx=3)
        self.sample_button = ttk.Button(controls, text='公开示例', command=lambda: self.set_image(read_image(ROOT/'assets/sample.jpg')))
        self.sample_button.pack(side='left', padx=3)
        self.benchmark_button = ttk.Button(controls, text='20 秒性能测量', command=self.start_benchmark)
        self.benchmark_button.pack(side='left', padx=3)
        area = ttk.Frame(root, padding=(22, 0, 22, 6)); area.pack(fill='both', expand=True)
        side = ttk.Frame(area, width=255, padding=(0, 0, 14, 0)); side.pack(side='left', fill='y'); side.pack_propagate(False)
        ttk.Label(side, text='实时特效', font=('Microsoft YaHei UI', 13, 'bold')).pack(anchor='w')
        self.effect = tk.StringVar(value='组合特效')
        for label in EFFECT_LABELS:
            ttk.Radiobutton(side, text=label, value=label, variable=self.effect, command=self.apply_settings).pack(anchor='w', pady=3)
        self.strength = tk.DoubleVar(value=.5)
        ttk.Label(side, text='美颜与口红强度').pack(anchor='w', pady=(12, 0))
        ttk.Scale(side, from_=0, to=1, variable=self.strength, command=lambda _: self.apply_settings()).pack(fill='x')
        self.landmarks = tk.BooleanVar(value=True)
        ttk.Checkbutton(side, text='显示检测框与五个关键点', variable=self.landmarks, command=self.apply_settings).pack(anchor='w', pady=9)
        ttk.Separator(side).pack(fill='x', pady=9)
        ttk.Button(side, text='选择两张图片进行验证', command=self.verify_images).pack(fill='x', pady=3)
        self.mesh_button = ttk.Button(side, text='当前画面 → 三维重建', command=lambda: self.process_current('mesh'))
        self.mesh_button.pack(fill='x', pady=3)
        ttk.Label(side, text='StarGAN 合成目标', font=('Microsoft YaHei UI', 11, 'bold')).pack(anchor='w', pady=(15, 3))
        self.hair = tk.StringVar(value='金色头发')
        ttk.Combobox(side, state='readonly', textvariable=self.hair, values=('黑色头发', '金色头发', '棕色头发')).pack(fill='x')
        self.male = tk.BooleanVar(value=False); self.young = tk.BooleanVar(value=True)
        ttk.Checkbutton(side, text='目标标签 Male', variable=self.male).pack(anchor='w')
        ttk.Checkbutton(side, text='目标标签 Young', variable=self.young).pack(anchor='w')
        self.gan_button = ttk.Button(side, text='当前画面 → 属性编辑', command=lambda: self.process_current('gan'))
        self.gan_button.pack(fill='x', pady=5)
        ttk.Label(side, text='标签指定合成方向，\n不是判断人物真实属性。', wraplength=235).pack(anchor='w')
        self.preview = tk.Label(area, bg='#162c25', fg='white', text='正在加载本机模型…', font=('Microsoft YaHei UI', 16))
        self.preview.pack(side='left', fill='both', expand=True)
        self.status = tk.StringVar(value='加载模型中')
        ttk.Label(root, textvariable=self.status, padding=(22, 8), wraplength=1090).pack(fill='x')
        ttk.Label(root, text='摄像头帧只在内存中处理；测量报告仅含统计数据。关闭窗口会释放摄像头。', padding=(22, 0, 22, 12), foreground='#5c7066').pack(fill='x')
        self.start_button.state(['disabled'])
        self.stop_button.state(['disabled'])
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.worker.start()
        root.after(15, self.poll)

    def send(self, command, value=None, new=True):
        if self.unavailable:
            self.status.set('模型工作线程已停止，请关闭并重新打开程序。')
            return
        if new:
            self.token += 1
            self.last_error = None
        self.worker.commands.put((command, self.token, value))

    def apply_settings(self):
        self.send('settings', (EFFECT_LABELS[self.effect.get()], self.strength.get(), self.landmarks.get()), new=False)

    def start_camera(self):
        if self.benchmark_start is not None:
            self.finish_benchmark()
        self.status.set('正在打开摄像头…')
        self.send('camera', int(self.camera_index.get()))

    def stop_camera(self):
        self.camera_active = False
        if self.benchmark_start is not None:
            self.finish_benchmark()
        self.send('stop')
        self.stop_button.state(['disabled'])

    def set_image(self, frame):
        if self.benchmark_start is not None:
            self.finish_benchmark()
        self.camera_active = False
        self.send('image', frame)

    def open_image(self):
        path = filedialog.askopenfilename(title='选择本机图片', filetypes=[('图片', '*.jpg *.jpeg *.png')])
        if path:
            try:
                self.set_image(read_image(path))
            except Exception as exc:
                messagebox.showerror('图片无法打开', str(exc), parent=self.root)

    def verify_images(self):
        paths = filedialog.askopenfilenames(title='选择两张各含一张清晰人脸的图片', filetypes=[('图片', '*.jpg *.jpeg *.png')])
        if not paths:
            return
        try:
            if len(paths) != 2:
                raise ValueError('请一次选择两张图片')
            images = (read_image(paths[0]), read_image(paths[1]), .363)
            self.camera_active = False
            if self.benchmark_start is not None:
                self.finish_benchmark()
            self.send('verify', images)
        except Exception as exc:
            messagebox.showerror('验证无法开始', str(exc), parent=self.root)

    def process_current(self, command):
        if self.latest is None:
            self.status.set('请先打开图片或开启摄像头。'); return
        self.camera_active = False
        if self.benchmark_start is not None:
            self.finish_benchmark()
        value = self.latest.copy()
        if command == 'gan':
            targets = [0, 0, 0, int(self.male.get()), int(self.young.get())]
            targets[('黑色头发', '金色头发', '棕色头发').index(self.hair.get())] = 1
            value = (value, targets)
        self.status.set('处理中，摄像头将暂停；窗口仍可操作…')
        self.send(command, value)

    def show(self, frame, label=None, size=None):
        label = self.preview if label is None else label
        maximum = size or (max(320, label.winfo_width()-12), max(240, label.winfo_height()-12))
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail(maximum, Image.Resampling.BILINEAR)
        photo = ImageTk.PhotoImage(image, master=self.root)
        label.configure(image=photo, text=''); label.image = photo

    def start_benchmark(self):
        if not self.camera_active:
            self.status.set('请先开启摄像头，再点击性能测量。'); return
        self.camera_samples = []
        self.benchmark_start = time.perf_counter()
        self.benchmark_button.state(['disabled'])

    def finish_benchmark(self, reason=None):
        elapsed = time.perf_counter()-self.benchmark_start
        self.benchmark_start = None
        samples = self.camera_samples
        path = ROOT/'reports/desktop-camera.json'
        path.parent.mkdir(exist_ok=True)
        report = {'frontend': 'native Tkinter; cv2.VideoCapture; no HTTP or browser',
                  'status': 'completed' if elapsed >= 20 and samples and reason is None else 'interrupted',
                  'interruption_reason': reason, 'requested_seconds': 20,
                  'seconds': elapsed, 'frames_rendered': len(samples),
                  'render_submission_fps': len(samples)/elapsed, 'device': self.camera_info,
                  'frames_with_face': sum(row['faces'] > 0 for row in samples),
                  'camera_frames_saved': False, 'effect': EFFECT_LABELS[self.effect.get()],
                  'scope': 'read + detection/effect + latest-frame queue + Tk PhotoImage assignment; excludes sensor exposure age and physical display scanout',
                  'frame_policy': 'latest completed frame only; old completed frames may be dropped to bound latency',
                  'settings_changed': len({(x['effect'], x['strength'], x['show_landmarks']) for x in samples}) > 1,
                  'latency_median_ms': float(np.median([x['latency_ms'] for x in samples])) if samples else None,
                  'latency_p95_ms': float(np.percentile([x['latency_ms'] for x in samples], 95, method='higher')) if samples else None,
                  'samples': samples}
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        self.benchmark_button.state(['!disabled'])
        self.status.set(f"测量已保存：{len(samples)} 帧 / {elapsed:.2f} 秒，{report['render_submission_fps']:.2f} FPS；reports/desktop-camera.json")

    def poll(self):
        if self.mesh_pipe is not None:
            try:
                while self.mesh_pipe.poll():
                    message = self.mesh_pipe.recv()
                    self.mesh_events.append(message)
                    if message['event'] == 'ready':
                        self.mesh_ready = True
                        self.status.set(f"原生 OpenGL 已绘制 {message['primitives_generated']:,} 个三角面。拖动鼠标或用左右键旋转；Home 归零；Esc 关闭。")
                    elif message['event'] == 'error':
                        self.last_error = '三维窗口：' + message['error']
                        self.status.set(self.last_error)
                    elif message['event'] == 'closed':
                        self.mesh_ready = False
            except (EOFError, OSError):
                self.mesh_pipe.close()
                self.mesh_pipe = None
                self.mesh_ready = False
        while not self.worker.events.empty():
            event, token, result = self.worker.events.get_nowait()
            if event not in ('ready', 'fatal') and token != self.token:
                continue
            self.received.append(event)
            if event == 'ready':
                self.ready = True
                self.start_button.state(['!disabled'])
                if not STARGAN_MODEL.exists(): self.gan_button.state(['disabled'])
                self.set_image(read_image(ROOT/'assets/sample.jpg'))
            elif event == 'camera':
                self.camera_active = True; self.camera_info = result
                self.stop_button.state(['!disabled'])
            elif event in ('stopped', 'image'):
                self.camera_active = False; self.stop_button.state(['disabled'])
                if event == 'stopped': self.status.set('摄像头已停止并释放；当前显示为最后一帧。')
            elif event == 'verify':
                self.last_result = result
                self.status.set(f"余弦相似度 {result['cosine']:.4f}，演示阈值 {result['threshold']}：{'通过' if result['match'] else '未通过'}。此结果不是身份认证。")
            elif event == 'gan':
                self.last_result = result
                joined = np.hstack([decode_image(result['input_crop']), decode_image(result['image'])])
                self.show(cv2.resize(joined, (768, 384), interpolation=cv2.INTER_NEAREST))
                self.status.set(f"左：输入裁剪；右：StarGAN 合成。模型推理 {result['model_ms']:.1f} ms，非实时视频特效。")
            elif event == 'mesh':
                self.mesh_result = result
                self.close_mesh()
                try:
                    from vision3d.native_viewer import start_viewer
                    self.mesh_process, self.mesh_pipe = start_viewer(result[0], result[2])
                    self.mesh_events = []
                    self.status.set(f'三维重建完成：{len(result[0]):,} 顶点 / {len(result[2]):,} 三角面；原生 OpenGL 窗口加载中。')
                except Exception as exc:
                    self.last_error = f'三维窗口无法启动：{exc}'
                    self.status.set(self.last_error)
            elif event in ('error', 'fatal'):
                self.camera_active = False; self.last_error = result
                # Invalidate a queued frame from the failed stream before it can
                # overwrite this error message during the same poll callback.
                self.token += 1
                if self.benchmark_start is not None:
                    self.finish_benchmark(reason=result)
                if event == 'fatal':
                    self.ready = False
                    self.unavailable = True
                    self.start_button.state(['disabled'])
                    result += '；模型线程已停止，请关闭并重新打开程序。'
                self.status.set(result)
                if not self.testing: messagebox.showerror('处理失败', result, parent=self.root)
        try:
            token, raw, result, metrics, started, camera = self.worker.frames.get_nowait()
            if token == self.token:
                self.latest = raw
                self.last_frame_metrics = metrics
                self.show(result)
                if camera and self.camera_active:
                    self.camera_info.update({'width': raw.shape[1], 'height': raw.shape[0]})
                    if self.benchmark_start is not None:
                        self.camera_samples.append({'latency_ms': (time.perf_counter()-started)*1000,
                                                    'pipeline_ms': metrics['pipeline_ms'], 'faces': metrics['faces'],
                                                    'effect': metrics['effect'], 'strength': metrics['strength'],
                                                    'show_landmarks': metrics['show_landmarks']})
                if self.benchmark_start is not None and time.perf_counter()-self.benchmark_start >= 20:
                    self.finish_benchmark()
                elif self.benchmark_start is not None:
                    self.status.set(f"正在测量 {time.perf_counter()-self.benchmark_start:.1f}/20 秒 · 人脸 {metrics['faces']} · 管线 {metrics['pipeline_ms']:.1f} ms")
                elif camera or not self.benchmark_start:
                    self.status.set(f"{'摄像头' if camera else '本机图片'} {raw.shape[1]}×{raw.shape[0]} · 人脸 {metrics['faces']} · 检测与特效 {metrics['pipeline_ms']:.1f} ms")
        except queue.Empty:
            pass
        self.root.after(15, self.poll)

    def close_mesh(self):
        if self.mesh_pipe is not None:
            try:
                self.mesh_pipe.send({'command': 'close'})
            except (BrokenPipeError, EOFError, OSError):
                pass
            self.mesh_pipe.close()
            self.mesh_pipe = None
        if self.mesh_process is not None:
            self.mesh_process.join(timeout=2)
            if self.mesh_process.is_alive():
                # Only this application's own spawned renderer is terminated.
                self.mesh_process.terminate()
                self.mesh_process.join(timeout=1)
            self.mesh_process = None
        self.mesh_ready = False

    def close(self):
        if self.benchmark_start is not None:
            self.finish_benchmark()
        self.worker.closing.set()
        self.close_mesh()
        self.worker.join(timeout=3)
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--self-test-no-camera', action='store_true', help='Verify native GUI and local models without reopening the camera')
    args = parser.parse_args()
    root = tk.Tk()
    testing = args.self_test or args.self_test_no_camera
    app = Desktop(root, testing=testing)
    if testing:
        from scripts.verify_desktop import run_checks
        root.after(100, lambda: run_checks(app, camera=not args.self_test_no_camera))
    root.mainloop()
    if testing:
        raise SystemExit(getattr(app, 'test_exit_code', 1))


if __name__ == '__main__':
    main()
