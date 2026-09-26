"""Real 3DDFA_V2 reconstruction: image -> 62 parameters -> 38,365 vertices.

Numerical pipeline adapted from cleardusk/3DDFA_V2 (MIT), revision recorded
in models/registry.json. Rendering uses matplotlib to avoid native compilers.
The input-specific prediction is a statistical 3DMM estimate, not a scan.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'models/3ddfa'


class NumpyOnlyUnpickler(pickle.Unpickler):
    """Read only verified upstream NumPy arrays, never arbitrary pickle code."""
    def find_class(self, module, name):
        allowed = {
            ('numpy', 'ndarray'): np.ndarray,
            ('numpy', 'dtype'): np.dtype,
            ('numpy.core.multiarray', '_reconstruct'): np._core.multiarray._reconstruct,
            ('numpy._core.multiarray', '_reconstruct'): np._core.multiarray._reconstruct,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f'Forbidden pickle global: {module}.{name}')
        return allowed[(module, name)]


def checked_asset(name):
    path = MODELS / name
    registry = json.loads((ROOT / 'models/registry.json').read_text(encoding='utf-8'))
    entry = next(e for e in registry['assets'] if e['path'] == path.relative_to(ROOT).as_posix())
    if not path.exists():
        raise FileNotFoundError('Run python scripts/fetch_models.py --with-3d first')
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        raise ValueError(f'SHA256 mismatch: {path.name}')
    return path


def load_array_pickle(name):
    with checked_asset(name).open('rb') as stream:
        return NumpyOnlyUnpickler(stream, encoding='latin1').load()


def export_onnx():
    import torch
    from .mobilenet_v1 import mobilenet
    model = mobilenet(num_classes=62).eval()
    checkpoint = torch.load(checked_asset('mb1_120x120.pth'), map_location='cpu', weights_only=True)['state_dict']
    # The released checkpoint also contains an unused auxiliary landmark head.
    state = {k.removeprefix('module.').replace('fc_param.', 'fc.'): v for k, v in checkpoint.items()
             if k.removeprefix('module.') not in {'fc_lm.weight', 'fc_lm.bias'}}
    model.load_state_dict(state, strict=True)
    output = MODELS / 'mb1_120x120.onnx'
    torch.onnx.export(model, torch.zeros(1, 3, 120, 120), str(output),
                      input_names=['input'], output_names=['output'],
                      opset_version=17, dynamo=False)
    return output


def reconstruct(image, bbox):
    """bbox=(x,y,width,height), BGR image; returns canonical/projected Nx3, Fx3."""
    import onnxruntime as ort
    x, y, w, h = map(float, bbox[:4])
    if min(w, h) <= 0 or image.ndim != 3:
        raise ValueError('Need a positive face box and a BGR image')
    size = max(1, int((w + h) / 2 * 1.58))
    cx, cy = x + w / 2, y + h / 2 + (w + h) / 2 * .14
    roi = [cx-size/2, cy-size/2, cx+size/2, cy+size/2]
    sx, sy, ex, ey = map(lambda n: int(round(n)), roi)
    crop = np.zeros((ey-sy, ex-sx, 3), np.uint8)
    x1, y1 = max(0, sx), max(0, sy)
    x2, y2 = min(image.shape[1], ex), min(image.shape[0], ey)
    if x2 <= x1 or y2 <= y1:
        raise ValueError('Face box is outside image')
    crop[y1-sy:y2-sy, x1-sx:x2-sx] = image[y1:y2, x1:x2]
    inp = cv2.resize(crop, (120, 120)).astype(np.float32).transpose(2, 0, 1)[None]
    inp = (inp - 127.5) / 128.0
    onnx_path = MODELS / 'mb1_120x120.onnx'
    if not onnx_path.exists():
        export_onnx()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(onnx_path), options, providers=['CPUExecutionProvider'])
    normal = load_array_pickle('param_mean_std_62d_120x120.pkl')
    start = time.perf_counter()
    param = session.run(None, {'input': inp})[0].flatten() * normal['std'] + normal['mean']
    inference_ms = (time.perf_counter() - start) * 1000
    bfm = load_array_pickle('bfm_noneck_v3.pkl')
    vertices = (bfm['u'].reshape(-1, 1) + bfm['w_shp'][:, :40] @ param[12:52, None]
                + bfm['w_exp'][:, :10] @ param[52:62, None]).reshape(3, -1, order='F')
    transform = param[:12].reshape(3, 4)
    projected = transform[:, :3] @ vertices + transform[:, 3:4]
    projected[0] = (projected[0] - 1) * size / 120 + roi[0]
    projected[1] = (120 - projected[1]) * size / 120 + roi[1]
    projected[2] = (projected[2] - 1) * size / 120
    projected[2] -= projected[2].min()
    triangles = np.asarray(load_array_pickle('tri.pkl'), dtype=np.int32).T
    assert triangles.ndim == 2 and triangles.shape[1] == 3
    assert triangles.min() >= 0 and triangles.max() < vertices.shape[1]
    assert np.isfinite(vertices).all() and np.isfinite(projected).all()
    return vertices.T, projected.T, triangles, param, inference_ms


def save_result(image, output, canonical, projected, triangles, param, inference_ms):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    # Canonical coordinates have arbitrary model scale, not millimetres.
    with (output / 'face.obj').open('w', encoding='utf-8') as f:
        f.write('# 3DDFA_V2 image-conditioned 3DMM reconstruction; arbitrary units\n')
        for v in canonical:
            f.write('v %.6f %.6f %.6f\n' % tuple(v))
        for t in triangles + 1:
            f.write('f %d %d %d\n' % tuple(t))
    overlay = image.copy()
    for x, y, z in projected[::20]:
        cv2.circle(overlay, (round(float(x)), round(float(y))), 1, (45, 230, 240), -1)
    cv2.imwrite(str(output / 'projection.jpg'), overlay)
    centered = canonical - canonical.mean(axis=0)
    centered /= np.ptp(centered, axis=0).max()
    fig = plt.figure(figsize=(12, 4), facecolor='#101726')
    for i, angle in enumerate((-40, 0, 40)):
        ax = fig.add_subplot(1, 3, i+1, projection='3d', facecolor='#101726')
        ax.plot_trisurf(centered[:, 0], centered[:, 2], centered[:, 1], triangles=triangles,
                        color='#53c9d2', linewidth=0, antialiased=False, shade=True)
        ax.view_init(elev=5, azim=-90 + angle)
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
        ax.set_title(f'Estimated face / yaw {angle:+d}', color='white')
    fig.subplots_adjust(top=.86, bottom=.02, left=0, right=1, wspace=0)
    fig.savefig(output / 'multiview.png', dpi=160)
    plt.close(fig)
    # Native browser WebGL is based on OpenGL ES; no extra graphics package.
    normals = np.zeros_like(centered)
    face_normals = np.cross(centered[triangles[:, 1]]-centered[triangles[:, 0]],
                            centered[triangles[:, 2]]-centered[triangles[:, 0]])
    for column in range(3):
        np.add.at(normals, triangles[:, column], face_normals)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    mesh_data = json.dumps({'vertices': centered.round(6).ravel().tolist(),
                            'normals': normals.round(5).ravel().tolist(),
                            'triangles': triangles.ravel().tolist()}, separators=(',', ':'))
    template = (Path(__file__).parent / 'viewer-template.html').read_text(encoding='utf-8')
    (output / 'viewer.html').write_text(template.replace('__MESH_DATA__', mesh_data), encoding='utf-8')
    metadata = {'method': '3DDFA_V2 pretrained MobileNet + learned shape/expression 3DMM',
                'vertices': len(canonical), 'triangles': len(triangles),
                'parameter_regression_ms_first_run': inference_ms, 'backend': 'ONNX Runtime CPU; 2 threads',
                'parameter_count': len(param), 'shape_coefficient_norm': float(np.linalg.norm(param[12:52])),
                'expression_coefficient_norm': float(np.linalg.norm(param[52:])),
                'metric_accuracy': 'Not measured: no paired 3D ground truth',
                'rendering': 'matplotlib static views + WebGL (OpenGL ES 2.0) interactive viewer',
                'limitations': 'Statistical monocular estimate, arbitrary scale, no accurate unseen back-of-head geometry.'}
    (output / 'reconstruction.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, default=ROOT / 'assets/sample.jpg')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/3d')
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error('Cannot read image')
    detector = cv2.FaceDetectorYN.create(str(ROOT / 'models/face_detection_yunet_2023mar.onnx'), '',
                                        (image.shape[1], image.shape[0]), score_threshold=.8)
    faces = detector.detect(image)[1]
    if faces is None:
        parser.error('No face found')
    face = max(faces, key=lambda face: float(face[2]*face[3]))
    result = reconstruct(image, face[:4])
    print(json.dumps(save_result(image, args.output, *result), indent=2))


if __name__ == '__main__':
    main()
