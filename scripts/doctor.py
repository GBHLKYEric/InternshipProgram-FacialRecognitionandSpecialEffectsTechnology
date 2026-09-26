"""Write a reproducible environment report without changing global settings."""
import ctypes
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def report():
    result = {'observed_at_utc': datetime.now(timezone.utc).isoformat(),
              'python': sys.version, 'executable': sys.executable,
              'platform': platform.platform(), 'logical_cpus': os.cpu_count(),
              'packages': {}, 'commands': {name: shutil.which(name) for name in ['git', 'docker', 'wsl', 'conda']}}
    for name in ['numpy', 'cv2', 'torch', 'torchvision', 'onnx', 'onnxruntime', 'scipy', 'matplotlib', 'PIL',
                 'jupyterlab', 'nbclient', 'nbformat', 'markdown', 'mmdet', 'mmcv', 'mediapipe']:
        try:
            module = importlib.import_module(name)
            result['packages'][name] = getattr(module, '__version__', 'installed')
            if name == 'torch':
                result['torch_cuda_available'] = module.cuda.is_available()
            if name == 'onnxruntime':
                result['onnx_providers'] = module.get_available_providers()
        except (ImportError, OSError) as exc:
            result['packages'][name] = f'unavailable: {exc}'
    if os.name == 'nt':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            result['cpu'] = winreg.QueryValueEx(key, 'ProcessorNameString')[0]
        kb = ctypes.c_ulonglong()
        if ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(kb)):
            result['installed_ram_gib'] = round(kb.value / 1024**2, 2)
        result['required_windows_environment_present'] = {k: bool(os.environ.get(k)) for k in ['SystemRoot', 'WINDIR', 'COMSPEC']}
        if shutil.which('wsl'):
            try:
                run = subprocess.run(['wsl', '-d', 'Ubuntu-24.04', '--', 'docker', '--version'],
                                     capture_output=True, timeout=20)
                result['wsl_docker'] = run.stdout.decode('utf-8', errors='replace').strip()
            except subprocess.TimeoutExpired:
                result['wsl_docker'] = 'Query timed out'
    registry = json.loads((ROOT / 'models/registry.json').read_text(encoding='utf-8'))
    result['assets'] = {entry['path']: (ROOT / entry['path']).is_file() and
                        hashlib.sha256((ROOT / entry['path']).read_bytes()).hexdigest() == entry['sha256']
                        for entry in registry['assets']}
    destination = ROOT / 'reports/environment.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == '__main__':
    report()
