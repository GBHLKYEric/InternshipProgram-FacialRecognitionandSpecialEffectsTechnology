"""Check that a camera failure ends an active benchmark and restores its button."""
import json
import queue
import tempfile
import time
import unittest
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import desktop


class DesktopFailureCheck(unittest.TestCase):
    def test_bad_verification_image_does_not_desynchronize_live_camera(self):
        app = desktop.Desktop.__new__(desktop.Desktop)
        app.camera_active = True
        app.benchmark_start = time.perf_counter()
        app.send = Mock(); app.finish_benchmark = Mock(); app.root = Mock()
        with patch.object(desktop.filedialog, 'askopenfilenames', return_value=['ok.jpg', 'bad.jpg']), \
             patch.object(desktop, 'read_image', side_effect=[np.zeros((2, 2, 3), np.uint8), ValueError('broken image')]), \
             patch.object(desktop.messagebox, 'showerror') as error:
            app.verify_images()
        self.assertTrue(app.camera_active)
        self.assertIsNotNone(app.benchmark_start)
        app.send.assert_not_called()
        app.finish_benchmark.assert_not_called()
        error.assert_called_once()

    def test_frame_processing_failure_keeps_worker_available_for_next_image(self):
        frame = np.zeros((2, 2, 3), np.uint8)
        vision = Mock()
        vision.process.side_effect = [RuntimeError('single frame failed'),
                                      (frame, {'pipeline_ms': .1, 'faces': 0})]
        camera = Mock()
        camera.isOpened.return_value = True
        camera.read.return_value = (True, frame)
        camera.getBackendName.return_value = 'test camera'
        worker = desktop.Worker()
        def wait_event(name):
            deadline = time.monotonic()+3
            while time.monotonic() < deadline:
                event = worker.events.get(timeout=3)
                if event[0] == name:
                    return event
            self.fail('missing event '+name)
        with patch.object(desktop, 'Vision', return_value=vision), \
             patch.object(desktop.cv2, 'VideoCapture', return_value=camera):
            worker.start()
            try:
                wait_event('ready')
                worker.commands.put(('camera', 1, 0))
                self.assertIn('single frame failed', wait_event('error')[2])
                self.assertTrue(worker.is_alive())
                self.assertTrue(worker.released)
                worker.commands.put(('image', 2, frame))
                wait_event('image')
                self.assertEqual(worker.frames.get(timeout=1)[0], 2)
            finally:
                worker.closing.set()
                worker.join(timeout=3)
        self.assertFalse(worker.is_alive())
        camera.release.assert_called_once()

    def test_camera_failure_finishes_as_interrupted_even_after_twenty_seconds(self):
        app = desktop.Desktop.__new__(desktop.Desktop)
        app.token = 3
        app.testing = True
        app.received = []
        app.mesh_pipe = None
        app.camera_active = True
        app.last_error = None
        app.benchmark_start = time.perf_counter() - 21
        app.camera_samples = [{'latency_ms': 50., 'pipeline_ms': 20., 'faces': 0,
                               'effect': 'all', 'strength': .5, 'show_landmarks': True}]
        app.camera_info = {'test': 'simulated failure only; no camera performance claim'}
        app.effect = SimpleNamespace(get=lambda: '组合特效')
        app.benchmark_button = Mock()
        app.status = Mock()
        app.root = Mock()
        events = queue.Queue(); events.put(('error', 3, 'simulated camera disconnect'))
        frames = queue.Queue()
        frames.put((3, None, None, {}, time.perf_counter(), True))
        app.show = Mock()
        app.worker = SimpleNamespace(events=events, frames=frames)
        with tempfile.TemporaryDirectory() as folder, patch.object(desktop, 'ROOT', Path(folder)):
            app.poll()
            report = json.loads((Path(folder)/'reports/desktop-camera.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'], 'interrupted')
            self.assertEqual(report['interruption_reason'], 'simulated camera disconnect')
        self.assertIsNone(app.benchmark_start)
        self.assertFalse(app.camera_active)
        self.assertEqual(app.token, 4)
        app.show.assert_not_called()
        app.status.set.assert_called_with('simulated camera disconnect')
        app.benchmark_button.state.assert_called_with(['!disabled'])
        app.root.after.assert_called_once()


if __name__ == '__main__':
    unittest.main()
