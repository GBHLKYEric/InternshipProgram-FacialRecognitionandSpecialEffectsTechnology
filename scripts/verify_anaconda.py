"""Verify a real conda prefix, CPU autograd, face detection and a Jupyter kernel."""
import argparse
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    assert (args.environment / 'conda-meta' / 'history').is_file(), 'Not a conda environment'
    assert os.path.samefile(sys.prefix, args.environment), 'Wrong interpreter'
    import cv2
    import jupyterlab
    import nbformat
    import numpy as np
    import torch
    import tkinter as tk
    from PIL import Image, ImageTk
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient

    window = tk.Tk()
    window.withdraw()
    try:
        tk_version = window.tk.call('info', 'patchlevel')
        photo = ImageTk.PhotoImage(Image.new('RGB', (8, 8)), master=window)
        label = tk.Label(window, image=photo)
        label.pack()
        window.update_idletasks()
        window.update()
        assert photo.width() == photo.height() == 8
    finally:
        window.destroy()

    x = torch.tensor([2.0], requires_grad=True)
    (x * x).sum().backward()
    assert x.grad.item() == 4.0
    image = cv2.imread(str(project / 'assets/sample.jpg'))
    assert image is not None, 'Run scripts/fetch_models.py first'
    detector = cv2.FaceDetectorYN.create(str(project / 'models/face_detection_yunet_2023mar.onnx'), '', (image.shape[1], image.shape[0]))
    _, faces = detector.detect(image)
    assert faces is not None and len(faces) >= 1
    with tempfile.TemporaryDirectory(prefix='face-conda-check-') as directory:
        kernels = Path(directory)
        spec = kernels / 'face-conda'
        spec.mkdir()
        (spec / 'kernel.json').write_text(json.dumps({'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'], 'display_name': 'Face conda verification', 'language': 'python'}), encoding='utf-8')
        nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('import sys, torch, cv2; print(sys.prefix); print(torch.arange(4).sum().item())')])
        manager = KernelManager(kernel_name='face-conda', kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
        try:
            NotebookClient(nb, km=manager, timeout=60).execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        output = ''.join(item.get('text', '') for item in nb.cells[0].outputs)
        assert '\n6\n' in output
    report = json.loads(args.report.read_text(encoding='utf-8-sig'))
    base = Path(report['installation']['physical_path'])
    report['anaconda_base'] = {
        'installer_metadata': json.loads((base / '.installer.info').read_text(encoding='utf-8')),
        'conda_version': subprocess.check_output([str(base / 'Scripts/conda.exe'), '--version'], text=True).strip(),
        'base_python': subprocess.check_output([str(base / 'python.exe'), '-c', 'import sys; print(sys.version)'], text=True).strip(),
    }
    report['status'] = 'installed_and_functionally_verified'
    report['verified_at_utc'] = datetime.now(timezone.utc).isoformat()
    report['validation'] = {'python': sys.version, 'executable': sys.executable, 'prefix': sys.prefix, 'torch': torch.__version__, 'torchvision': metadata.version('torchvision'), 'opencv': cv2.__version__, 'numpy': np.__version__, 'pillow': metadata.version('Pillow'), 'jupyterlab': jupyterlab.__version__, 'torch_fidelity': metadata.version('torch-fidelity'), 'cpu_autograd': True, 'sample_faces': len(faces), 'tk_patchlevel': tk_version, 'tk_window_create_update_destroy': True, 'pillow_imagetk_binding': True, 'camera_opened': False, 'notebook_kernel_execution': True, 'notebook_output': output.strip()}
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report['validation'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
