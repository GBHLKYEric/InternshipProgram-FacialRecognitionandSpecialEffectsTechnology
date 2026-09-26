"""Read installed GPU/runtime capability; never installs drivers or packages."""
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

import cv2
import numpy as np
import onnxruntime as ort
import torch


def main():
    project = Path(__file__).resolve().parents[1]
    cv2.setNumThreads(2)
    cv2.ocl.setUseOpenCL(True)
    test = np.ones((128, 128), dtype=np.float32)
    blurred = cv2.GaussianBlur(cv2.UMat(test), (3, 3), 1).get()
    device = cv2.ocl.Device_getDefault()
    report = {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'method': 'Read installed runtimes; small local UMat operation and YuNet inference. No package/driver/environment modification.',
        'torch': {'version': torch.__version__, 'xpu_compiled': torch.xpu._is_compiled(), 'xpu_available': torch.xpu.is_available(), 'cuda_available': torch.cuda.is_available()},
        'onnxruntime': {'version': ort.__version__, 'providers': ort.get_available_providers()},
        'opencv': {'version': cv2.__version__, 'opencl_available': cv2.ocl.haveOpenCL(), 'opencl_enabled': cv2.ocl.useOpenCL(), 'device_name': device.name(), 'device_version': device.version(), 'driver_version': device.driverVersion(), 'umat_blur_result_correct': bool(np.allclose(blurred, test)), 'dnn_opencv_available_targets': list(map(int, cv2.dnn.getAvailableTargets(cv2.dnn.DNN_BACKEND_OPENCV)))},
        'sources': {
            'pytorch_xpu': 'https://github.com/pytorch/pytorch/blob/main/docs/source/notes/get_start_xpu.md',
            'openvino': 'https://docs.openvino.ai/2026/about-openvino/release-notes-openvino/system-requirements.html'
        },
        'limitations': [
            'The installed CPU-only PyTorch wheel cannot use XPU. This is not evidence that the hardware is unsupported.',
            'Official PyTorch XPU documentation lists Arrow Lake-H under Windows 11. An isolated XPU wheel installation and autograd test are still needed before claiming PyTorch GPU training works here.',
            'Requested OpenCV DNN target is not proof of execution on that device. OpenCV 5 new graph engine warns that targets are not supported; both timings may therefore be CPU execution.',
            'OpenVINO is not installed or tested in this environment.',
            'This tiny sequential local measurement is not a controlled performance benchmark and does not demonstrate a general speedup.'
        ]
    }
    image = cv2.imread(str(project / 'assets/sample.jpg'))
    assert image is not None
    report['yunet'] = {}
    report['yunet_execution_device_verified'] = False
    for name, target in [('cpu_requested', cv2.dnn.DNN_TARGET_CPU), ('opencl_requested', cv2.dnn.DNN_TARGET_OPENCL)]:
        try:
            detector = cv2.FaceDetectorYN.create(str(project / 'models/face_detection_yunet_2023mar.onnx'), '', (image.shape[1], image.shape[0]), 0.8, 0.3, 5000, cv2.dnn.DNN_BACKEND_OPENCV, target)
            for _ in range(3):
                detector.detect(image)
            elapsed = []
            for _ in range(10):
                start = time.perf_counter()
                _, faces = detector.detect(image)
                elapsed.append((time.perf_counter() - start) * 1000)
            report['yunet'][name] = {'detected_faces': 0 if faces is None else len(faces), 'median_ms': statistics.median(elapsed), 'samples': len(elapsed), 'input_size': [image.shape[1], image.shape[0]], 'opencv_cpu_threads': cv2.getNumThreads(), 'output': None if faces is None else faces.tolist()}
        except cv2.error as error:
            report['yunet'][name] = {'error': str(error)}
    destination = project / 'reports/gpu-capability.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
