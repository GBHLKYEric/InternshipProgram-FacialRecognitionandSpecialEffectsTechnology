"""Small, meaningful check: real reconstructed geometry and safe asset loading."""
import io
import json
import pickle
import cv2
import numpy as np
from .reconstruct import ROOT, NumpyOnlyUnpickler, reconstruct


def main():
    # An unknown global must be rejected before it can be constructed/called.
    try:
        NumpyOnlyUnpickler(io.BytesIO(b'cos\nsystem\n.')).load()
    except pickle.UnpicklingError:
        pass
    else:
        raise AssertionError('Unsafe pickle global was accepted')
    image = cv2.imread(str(ROOT / 'assets/sample.jpg'))
    detector = cv2.FaceDetectorYN.create(str(ROOT / 'models/face_detection_yunet_2023mar.onnx'), '',
                                        (image.shape[1], image.shape[0]))
    face = detector.detect(image)[1][0]
    vertices, projected, faces, params, _ = reconstruct(image, face[:4])
    assert vertices.shape == projected.shape == (38365, 3)
    assert faces.shape == (76073, 3) and len(params) == 62
    assert np.ptp(vertices[:, 2]) > 0 and np.isfinite(vertices).all()
    mirror = cv2.flip(image, 1)
    mirrored_face = detector.detect(mirror)[1][0]
    _, _, _, mirror_params, _ = reconstruct(mirror, mirrored_face[:4])
    assert not np.allclose(params, mirror_params), 'Output must depend on input image'
    result = {'safe_pickle': True, 'finite_3d_geometry': True, 'input_dependent': True,
              'vertices': len(vertices), 'triangles': len(faces)}
    (ROOT / 'reports/3d/selfcheck.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result)


if __name__ == '__main__':
    main()
