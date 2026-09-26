"""Native OpenGL 3.3+ full-mesh viewer, isolated from Tk by multiprocessing spawn.

Dependency: pyglet 2.1.16, BSD-3-Clause; see licenses/pyglet-2.1.16-LICENSE.txt.
The shader and IPC implementation are original project code (project MIT license).
No browser, camera, driver installation, or automatic image saving is involved.

Use start_viewer(vertices, triangles) from a guarded application entry point.
It returns (Process, Connection); poll the connection from Tk.after, never block
Tk waiting for a message. Send dictionaries:
    {'command': 'yaw', 'degrees': 35}
    {'command': 'capture', 'path': '/absolute/path/image.png'}
    {'command': 'close'}
Events: ready, rotated, captured, closed, error. Images are saved only on an
explicit capture command. Report errors to the user if no GL context is available.
All mesh buffers and the GL context live only in the spawned child process.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import time
import traceback

import numpy as np

VERTEX_SHADER = """#version 330 core
in vec3 position;
in vec3 normal;
uniform float angle;
uniform float aspect;
out vec3 surface_normal;
void main() {
    float c = cos(angle), s = sin(angle);
    mat3 rotation = mat3(c, 0., -s, 0., 1., 0., s, 0., c);
    vec3 p = rotation * position;
    surface_normal = rotation * normal;
    float scale = 1.72 * min(1., aspect);
    // Canonical 3DMM faces point toward +Z; larger Z is nearer the camera.
    gl_Position = vec4(scale * p.x / aspect, scale * p.y, -p.z * .5, 1.);
}
"""
FRAGMENT_SHADER = """#version 330 core
in vec3 surface_normal;
out vec4 final_color;
void main() {
    vec3 n = normalize(surface_normal);
    vec3 key = normalize(vec3(-.45, .6, 1.));
    // Two-sided shading accommodates upstream winding without culling faces.
    float light = .23 + .77 * abs(dot(n, key));
    final_color = vec4(vec3(.22, .77, .82) * light, 1.);
}
"""


def prepare_mesh(vertices, triangles):
    """Validate every input index and return centered, uniformly scaled buffers."""
    points = np.asarray(vertices, dtype=np.float32)
    source_indices = np.asarray(triangles)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError('vertices must have shape (N, 3), N >= 3')
    if not np.isfinite(points).all():
        raise ValueError('vertices contain non-finite coordinates')
    if source_indices.ndim != 2 or source_indices.shape[1] != 3 or not len(source_indices):
        raise ValueError('triangles must have shape (F, 3), F >= 1')
    if not np.issubdtype(source_indices.dtype, np.integer):
        raise ValueError('triangle indices must be integers')
    if source_indices.min() < 0 or source_indices.max() >= len(points):
        raise ValueError('triangle index is outside the vertex buffer')
    indices = np.ascontiguousarray(source_indices, dtype=np.uint32)
    points = points - points.mean(axis=0)
    scale = float(np.ptp(points, axis=0).max())
    if scale <= 1e-12:
        raise ValueError('mesh has zero extent')
    points = np.ascontiguousarray(points / scale, dtype=np.float32)
    face_normals = np.cross(points[indices[:, 1]] - points[indices[:, 0]],
                            points[indices[:, 2]] - points[indices[:, 0]])
    normals = np.zeros_like(points)
    for column in range(3):
        np.add.at(normals, indices[:, column], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(lengths, 1e-12)
    normals[lengths[:, 0] <= 1e-12] = (0, 0, 1)
    return points, indices, np.ascontiguousarray(normals)


def start_viewer(vertices, triangles, *, title='3DMM face estimate - drag to rotate',
                 width=800, height=720, visible=True):
    """Start asynchronously; no pyglet or OpenGL is imported in the Tk process.

    Caller must retain both return values, drain events, send close when the Tk
    parent closes, and join the child without blocking Tk for a long interval.
    The application entry point must use ``if __name__ == '__main__':``.
    """
    points, indices, normals = prepare_mesh(vertices, triangles)
    if not 64 <= width <= 4096 or not 64 <= height <= 4096:
        raise ValueError('window dimensions must be between 64 and 4096 pixels')
    context = mp.get_context('spawn')
    parent, child = context.Pipe(duplex=True)
    process = context.Process(target=_run, args=(child, points, indices, normals,
                              str(title), int(width), int(height), bool(visible)),
                              name='FaceVision-OpenGL', daemon=True)
    try:
        process.start()
    except BaseException:
        parent.close()
        child.close()
        raise
    child.close()
    return process, parent


def _run(connection, points, indices, normals, title, width, height, visible):
    window = vertex_list = program = None
    closed = False

    def emit(event, **values):
        try:
            connection.send({'event': event, **values})
        except (BrokenPipeError, EOFError, OSError):
            pass

    try:
        import ctypes
        import pyglet
        pyglet.options['shadow_window'] = False
        from pyglet import gl
        from pyglet.graphics.shader import Shader, ShaderProgram

        config = gl.Config(double_buffer=True, depth_size=24, major_version=3,
                           minor_version=3)
        window = pyglet.window.Window(width=width, height=height, caption=title,
                                      resizable=True, visible=visible,
                                      config=config, vsync=True)
        window.switch_to()
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LESS)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glClearColor(12 / 255, 23 / 255, 37 / 255, 1.)
        program = ShaderProgram(Shader(VERTEX_SHADER, 'vertex'),
                                Shader(FRAGMENT_SHADER, 'fragment'))
        vertex_list = program.vertex_list_indexed(len(points), gl.GL_TRIANGLES,
                      indices.ravel().tolist(), position=('f', points.ravel()),
                      normal=('f', normals.ravel()))
        yaw = 0.
        generated_primitives = None
        frame_count = 0

        def render(measure=False):
            nonlocal generated_primitives, frame_count
            window.switch_to()
            fb_width, fb_height = window.get_framebuffer_size()
            gl.glViewport(0, 0, fb_width, fb_height)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            program.use()
            program['angle'] = math.radians(yaw)
            program['aspect'] = fb_width / max(1, fb_height)
            query = gl.GLuint()
            if measure:
                gl.glGenQueries(1, ctypes.byref(query))
                gl.glBeginQuery(gl.GL_PRIMITIVES_GENERATED, query)
            vertex_list.draw(gl.GL_TRIANGLES)
            if measure:
                gl.glEndQuery(gl.GL_PRIMITIVES_GENERATED)
                result = gl.GLuint()
                gl.glGetQueryObjectuiv(query, gl.GL_QUERY_RESULT, ctypes.byref(result))
                generated_primitives = result.value
                gl.glDeleteQueries(1, ctypes.byref(query))
            program.stop()
            error = gl.glGetError()
            if error != gl.GL_NO_ERROR:
                raise RuntimeError(f'OpenGL drawing error: {error}')
            frame_count += 1

        def pose():
            return {'degrees': yaw, 'shader_radians': float(program['angle']),
                    'frame_count': frame_count, 'triangles': len(indices),
                    'vertices': len(points)}

        def close_viewer(reason):
            nonlocal closed
            if not closed:
                closed = True
                emit('closed', reason=reason)
                pyglet.app.exit()

        def set_yaw(value):
            nonlocal yaw
            value = float(value)
            if not math.isfinite(value):
                raise ValueError('yaw must be finite')
            yaw = max(-180., min(180., value))
            window.set_caption(f'{title} | {len(indices):,} faces | yaw {yaw:+.1f} deg')

        @window.event
        def on_draw():
            render()

        @window.event
        def on_close():
            close_viewer('window')
            return pyglet.event.EVENT_HANDLED

        @window.event
        def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
            if buttons & pyglet.window.mouse.LEFT:
                set_yaw(yaw + dx * .4)

        @window.event
        def on_key_press(symbol, modifiers):
            key = pyglet.window.key
            if symbol == key.ESCAPE:
                close_viewer('escape')
            elif symbol == key.LEFT:
                set_yaw(yaw - 5)
            elif symbol == key.RIGHT:
                set_yaw(yaw + 5)
            elif symbol == key.HOME:
                set_yaw(0)

        def commands(_dt):
            if closed:
                return
            try:
                if not connection.poll():
                    return
                request = connection.recv()
                command = request.get('command')
                if command == 'close':
                    close_viewer('command')
                elif command == 'yaw':
                    set_yaw(request['degrees'])
                    render()
                    gl.glFinish()
                    window.flip()
                    emit('rotated', **pose())
                elif command == 'capture':
                    path = Path(request['path'])
                    if not path.is_absolute() or path.suffix.lower() != '.png':
                        raise ValueError('capture requires an absolute .png path')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    render(measure=True)
                    gl.glFinish()
                    gl.glReadBuffer(gl.GL_BACK)
                    pyglet.image.get_buffer_manager().get_color_buffer().save(str(path))
                    window.flip()
                    emit('captured', path=str(path), bytes=path.stat().st_size,
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         primitives_generated=generated_primitives, **pose())
                else:
                    raise ValueError(f'Unknown viewer command: {command!r}')
            except (EOFError, BrokenPipeError, OSError) as error:
                # Broken connections mean the parent is gone. File errors can be
                # reported while retaining the window and the existing mesh.
                if isinstance(error, (EOFError, BrokenPipeError)):
                    close_viewer('parent_disconnected')
                else:
                    emit('error', error=str(error), fatal=False)
            except Exception as error:
                emit('error', error=str(error), fatal=False)

        render(measure=True)
        gl.glFinish()
        window.flip()
        emit('ready', pyglet=pyglet.version, vendor=gl.gl_info.get_vendor(),
             renderer=gl.gl_info.get_renderer(), opengl=list(gl.gl_info.get_version()),
             depth_bits=window.config.depth_size,
             framebuffer=list(window.get_framebuffer_size()),
             primitives_generated=generated_primitives,
             automatic_capture=False, **pose())
        pyglet.clock.schedule_interval(commands, 1 / 60)
        pyglet.app.run(interval=1 / 30)
    except Exception as error:
        emit('error', error=str(error), traceback=traceback.format_exc(), fatal=True)
    finally:
        if window is not None:
            try:
                window.switch_to()
                if vertex_list is not None:
                    vertex_list.delete()
                if program is not None:
                    program.delete()
                window.close()
            except Exception:
                pass
        connection.close()


def _receive(connection, event, timeout=45):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if connection.poll(min(.1, max(0., deadline - time.monotonic()))):
            message = connection.recv()
            if message['event'] == 'error':
                raise RuntimeError(message)
            if message['event'] == event:
                return message
    raise TimeoutError(f'No native viewer {event!r} event within {timeout} seconds')


def self_test(output):
    """Use only the bundled public NASA image; no camera or personal images."""
    import cv2
    from PIL import Image
    from .reconstruct import reconstruct
    root = Path(__file__).resolve().parents[1]
    sample = root / 'assets/sample.jpg'
    image = cv2.imread(str(sample))
    if image is None:
        raise FileNotFoundError('Run scripts/fetch_models.py first')
    detector = cv2.FaceDetectorYN.create(str(root / 'models/face_detection_yunet_2023mar.onnx'),
                                       '', (image.shape[1], image.shape[0]))
    faces = detector.detect(image)[1]
    if faces is None:
        raise RuntimeError('No face in the public example')
    vertices, _, triangles, _, _ = reconstruct(image, max(faces, key=lambda f: f[2]*f[3])[:4])
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    process, pipe = start_viewer(vertices, triangles, title='NASA sample - native OpenGL validation')
    report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'sample': 'assets/sample.jpg (NASA astronaut example)',
              'sample_sha256': hashlib.sha256(sample.read_bytes()).hexdigest(),
              'camera_opened': False, 'mesh_type': '3DDFA_V2 image-conditioned 3DMM estimate; arbitrary units',
              'process_start_method': 'spawn', 'screenshots_explicitly_requested': True,
              'limitations': ['OpenGL rendering validation, not a reconstruction accuracy evaluation.',
                             'No end-to-end speed benchmark or physical face measurements.']}
    try:
        report['ready'] = _receive(pipe, 'ready')
        assert report['ready']['primitives_generated'] == len(triangles) == 76073
        assert report['ready']['vertices'] == 38365
        pipe.send({'command': 'capture', 'path': str(output / 'yaw-000.png')})
        report['capture_0'] = _receive(pipe, 'captured')
        pipe.send({'command': 'yaw', 'degrees': 35})
        report['rotation'] = _receive(pipe, 'rotated')
        assert abs(report['rotation']['shader_radians'] - math.radians(35)) < 1e-6
        pipe.send({'command': 'capture', 'path': str(output / 'yaw-035.png')})
        report['capture_35'] = _receive(pipe, 'captured')
        baseline = np.asarray(Image.open(output / 'yaw-000.png').convert('RGB'))
        rotated = np.asarray(Image.open(output / 'yaw-035.png').convert('RGB'))
        assert baseline.shape == rotated.shape
        diff = np.abs(baseline.astype(np.int16) - rotated.astype(np.int16))
        changed = float(np.mean(np.any(diff > 5, axis=2)))
        background = np.array([12, 23, 37])
        foreground_fraction = float(np.mean(np.any(np.abs(baseline.astype(np.int16)-background) > 5, axis=2)))
        assert changed > .03 and foreground_fraction > .08, 'Expected a visible, rotated mesh'
        report['pixel_check'] = {'size': list(baseline.shape), 'mean_absolute_difference_0_255': float(diff.mean()),
                                 'changed_pixel_fraction_over_5': changed, 'baseline_foreground_fraction': foreground_fraction}
        pipe.send({'command': 'yaw', 'degrees': -35})
        report['rotation_minus_35'] = _receive(pipe, 'rotated')
        assert abs(report['rotation_minus_35']['shader_radians'] - math.radians(-35)) < 1e-6
        pipe.send({'command': 'capture', 'path': str(output / 'yaw-minus035.png')})
        report['capture_minus_35'] = _receive(pipe, 'captured')
        assert report['capture_minus_35']['primitives_generated'] == 76073
        negative = np.asarray(Image.open(output / 'yaw-minus035.png').convert('RGB'))
        assert np.any(negative != baseline) and np.any(negative != rotated)
        report['distinct_rendered_yaws'] = [0, 35, -35]
        pipe.send({'command': 'yaw', 'degrees': 0})
        report['reset'] = _receive(pipe, 'rotated')
        pipe.send({'command': 'capture', 'path': str(output / 'yaw-000-reset.png')})
        report['capture_reset'] = _receive(pipe, 'captured')
        reset = np.asarray(Image.open(output / 'yaw-000-reset.png').convert('RGB'))
        report['reset_identical_pixels'] = bool(np.array_equal(baseline, reset))
        assert report['reset_identical_pixels']
        pipe.send({'command': 'close'})
        report['closed'] = _receive(pipe, 'closed')
        process.join(timeout=10)
        report['child_exitcode'] = process.exitcode
        assert not process.is_alive() and process.exitcode == 0
        report['status'] = 'verified'
    except Exception:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        if process.is_alive():
            try:
                pipe.send({'command': 'close'})
                process.join(timeout=3)
            except (BrokenPipeError, EOFError, OSError):
                pass
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
        pipe.close()
        (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    mp.freeze_support()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true', help='Validate with the public NASA sample and explicit PNG captures')
    parser.add_argument('--output', type=Path, default=Path('reports/3d-native'))
    args = parser.parse_args()
    if not args.self_test:
        parser.error('Use --self-test, or import start_viewer from the desktop application')
    print(json.dumps(self_test(args.output), ensure_ascii=False, indent=2))
