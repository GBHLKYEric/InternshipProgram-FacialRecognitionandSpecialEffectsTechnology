"""Exercise real Tk callbacks, camera lifecycle and optional local models.

Invoked by desktop.py --self-test. Captures only the public sample app window;
never saves webcam frames. The JSON distinguishes render submission from display.
"""
from __future__ import annotations
import json
import time
from PIL import ImageGrab

from app import ROOT, STARGAN_MODEL


def run_checks(app, camera=True):
    from desktop import EFFECT_LABELS, read_image
    import numpy as np
    report = {'frontend': 'Tkinter native window; direct OpenCV VideoCapture',
              'checks': [], 'errors': [], 'webcam_images_saved': False,
              'camera_test_in_this_run': camera}
    steps = []
    deadline = time.monotonic() + 150

    def record(name):
        report['checks'].append(name)
        print('PASS:', name, flush=True)

    def later(predicate, action, label):
        steps.append((predicate, action, label))

    later(lambda: app.ready and app.latest is not None,
          lambda: record('real Tk window mapped, Vision loaded, public image rendered'), 'initialization')
    for label in EFFECT_LABELS:
        def set_effect(label=label):
            app.effect.set(label); app.apply_settings()
        later(lambda: True, set_effect, 'effect '+label)
        later(lambda label=label: app.worker.settings[0] == EFFECT_LABELS[label],
              lambda label=label: record('native effect control '+label), 'effect accepted '+label)
    later(lambda: True, lambda: app.send('verify', (app.latest.copy(), app.latest.copy(), .363)), 'verify invoke')
    later(lambda: app.last_result is not None and 'cosine' in app.last_result,
          lambda: (record('same public image verification') if abs(app.last_result['cosine']-1) < 1e-5 else (_ for _ in ()).throw(AssertionError('cosine'))), 'verify result')
    if STARGAN_MODEL.exists():
        def gan_checked():
            record('StarGAN from current image in native preview')
            app.root.update_idletasks()
            ImageGrab.grab(window=int(app.root.wm_frame(), 16)).save(ROOT/'reports/desktop-attributes.png')
        later(lambda: True, lambda: app.gan_button.invoke(), 'GAN button')
        later(lambda: app.last_result is not None and 'model_ms' in app.last_result,
              gan_checked, 'GAN result')
    else:
        report['stargan'] = 'not tested: local export absent at test time'
    if (ROOT/'models/3ddfa/bfm_noneck_v3.pkl').exists():
        later(lambda: True, lambda: app.mesh_button.invoke(), 'mesh button')
        def event(name):
            return next((e for e in reversed(app.mesh_events) if e['event'] == name), None)
        def mesh_checked():
            ready = event('ready')
            assert ready['primitives_generated'] == 76073
            assert ready['vertices'] == 38365
            report['opengl'] = ready
            record('desktop button opened native OpenGL and rendered all 76073 triangles')
            app.mesh_pipe.send({'command': 'yaw', 'degrees': 35})
        later(lambda: app.mesh_ready, mesh_checked, 'full mesh rendered')
        def mesh_rotated():
            import math
            assert abs(event('rotated')['shader_radians']-math.radians(35)) < 1e-6
            report['opengl_rotation'] = event('rotated')
            record('35 degree rotation acknowledged after actual GPU draw')
            app.mesh_pipe.send({'command': 'capture', 'path': str(ROOT/'reports/desktop-3d.png')})
        later(lambda: event('rotated') is not None, mesh_rotated, 'mesh rotation')
        def mesh_capture():
            assert event('captured')['primitives_generated'] == 76073
            report['opengl_capture'] = event('captured')
            process = app.mesh_process
            app.close_mesh()
            assert process.exitcode == 0
            record('public sample full mesh captured; native child closed with exit code 0')
        later(lambda: event('captured') is not None, mesh_capture, 'mesh screenshot')
    if camera:
        later(lambda: True, lambda: app.start_button.invoke(), 'camera start button')
        later(lambda: app.camera_active, lambda: app.benchmark_button.invoke(), 'camera opened')
        later(lambda: len(app.camera_samples) > 0 and app.benchmark_start is None,
              lambda: (record('20 second native camera capture/process/Tk render submission'), app.stop_button.invoke()), '20 second camera benchmark')
        later(lambda: app.worker.released and 'stopped' in app.received,
              lambda: (record('stop button released camera'), app.start_button.invoke()), 'camera release')
        later(lambda: app.camera_active,
              lambda: (record('camera successfully reopened after release'), app.stop_button.invoke()), 'camera reopen')
    later(lambda: app.worker.released,
          lambda: app.sample_button.invoke(), 'restore public sample')
    later(lambda: not app.camera_active and app.latest is not None and app.latest.shape[:2] == read_image(ROOT/'assets/sample.jpg').shape[:2],
          lambda: record('public sample restored before screenshot'), 'public sample')

    def finish():
        try:
            app.root.update_idletasks()
            assert not app.camera_active and app.worker.released
            assert np.array_equal(app.latest, read_image(ROOT/'assets/sample.jpg'))
            # Capture only this application's native window after restoring a
            # public sample. No desktop-wide or camera-image screenshot is saved.
            handle = int(app.root.wm_frame(), 16)
            ImageGrab.grab(window=handle).save(ROOT/'reports/desktop-window.png')
            record('native window screenshot contains public sample only')
            report['window'] = {'width': app.root.winfo_width(), 'height': app.root.winfo_height(),
                                'mapped': bool(app.root.winfo_ismapped()), 'tk': app.root.tk.call('info', 'patchlevel')}
        except Exception as exc:
            report['errors'].append(f'{type(exc).__name__}: {exc}')
        app.close()
        report['camera_released_on_close'] = app.worker.released and not app.worker.is_alive()
        report['passed'] = not report['errors'] and report['camera_released_on_close']
        name = 'desktop-verification.json' if camera else 'desktop-integration.json'
        (ROOT/'reports'/name).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        app.test_exit_code = 0 if report['passed'] else 1
        print(json.dumps(report, ensure_ascii=False), flush=True)

    def advance():
        try:
            if app.last_error:
                raise RuntimeError(app.last_error)
            if time.monotonic() > deadline:
                raise TimeoutError(steps[0][2] if steps else 'finish')
            if not steps:
                finish(); return
            predicate, action, label = steps[0]
            if predicate():
                action(); steps.pop(0)
        except Exception as exc:
            report['errors'].append(f'{type(exc).__name__}: {exc}')
            app.stop_camera()
            app.sample_button.invoke()
            app.root.after(800, finish)
            return
        app.root.after(100, advance)
    advance()
