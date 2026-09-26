"""Local face detection, explicit pair verification, and effects. Run: python app.py."""
from __future__ import annotations

import argparse
import base64
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
EFFECTS = {"none", "glasses", "crown", "beauty", "lipstick", "all"}
STARGAN_MODEL = ROOT / "runs/stargan-deploy/generator.onnx"


def decode_image(value: str) -> np.ndarray:
    if not isinstance(value, str) or not value.startswith("data:image/"):
        raise ValueError("请提供图像文件")
    try:
        raw = base64.b64decode(value.split(",", 1)[1], validate=True)
    except (ValueError, IndexError) as exc:
        raise ValueError("图像编码无效") from exc
    if len(raw) > 8_000_000:
        raise ValueError("图像不得超过 8 MB")
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("无法读取此图像，请使用 JPEG 或 PNG")
    if max(frame.shape[:2]) > 4096:
        raise ValueError("图像边长不得超过 4096 像素")
    ratio = min(1.0, 960 / max(frame.shape[:2]))
    return cv2.resize(frame, None, fx=ratio, fy=ratio) if ratio < 1 else frame


def encode_image(frame: np.ndarray) -> str:
    ok, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ValueError("图像编码失败")
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


class Vision:
    def __init__(self, model_dir: Path = ROOT / "models"):
        detector = next(model_dir.glob("*yunet*.onnx"), None)
        recognizer = next(model_dir.glob("*sface*.onnx"), None)
        if detector is None or recognizer is None:
            raise FileNotFoundError("缺少模型，请先执行 python scripts/fetch_models.py")
        cv2.setNumThreads(4)
        self.detector = cv2.FaceDetectorYN.create(str(detector), "", (320, 320), .8, .3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer), "")
        self.stargan = None

    def detect(self, frame: np.ndarray) -> np.ndarray:
        self.detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = self.detector.detect(frame)
        return faces if faces is not None else np.empty((0, 15), dtype=np.float32)

    def feature(self, frame: np.ndarray) -> np.ndarray:
        faces = self.detect(frame)
        if len(faces) != 1:
            raise ValueError(f"验证图像须恰好有 1 张清晰人脸；当前检测到 {len(faces)} 张")
        aligned = self.recognizer.alignCrop(frame, faces[0])
        return self.recognizer.feature(aligned)

    def verify(self, first: np.ndarray, second: np.ndarray, threshold: float) -> dict:
        a, b = self.feature(first), self.feature(second)
        score = float(np.clip(self.recognizer.match(a, b, cv2.FaceRecognizerSF_FR_COSINE), -1, 1))
        return {"cosine": score, "threshold": threshold, "match": score >= threshold,
                "note": "阈值演示结果，不是身份认证或 LFW 准确率；应在独立验证集校准。"}

    def edit_attributes(self, frame: np.ndarray, targets: list) -> dict:
        if not isinstance(targets, list) or len(targets) != 5 or any(type(v) is not int or v not in (0, 1) for v in targets):
            raise ValueError("属性目标须依次为 Black_Hair、Blond_Hair、Brown_Hair、Male、Young 的五个0/1整数")
        if sum(targets[:3]) != 1:
            raise ValueError("请选择一种目标发色")
        faces = self.detect(frame)
        if len(faces) != 1:
            raise ValueError(f"属性编辑须恰好有1张清晰人脸；当前检测到{len(faces)}张")
        if not STARGAN_MODEL.is_file():
            raise ValueError("本机尚未导出StarGAN模型，请先完成研究训练与部署导出")
        if self.stargan is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise ValueError("属性编辑需要研究环境中的onnxruntime依赖") from exc
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            self.stargan = ort.InferenceSession(str(STARGAN_MODEL), sess_options=options, providers=['CPUExecutionProvider'])
        x, y, width, height = faces[0, :4]
        side = max(2, int(np.ceil(max(width, height) * 1.8)))
        center_x, center_y = x + width / 2, y + height * .45
        # ponytail: a detector-guided square crop is not CelebA alignment; large
        # poses/backgrounds can reduce editing quality. No whole-frame compositing.
        padded = cv2.copyMakeBorder(frame, side, side, side, side, cv2.BORDER_REFLECT_101)
        left, top = int(round(center_x - side/2)) + side, int(round(center_y - side/2)) + side
        crop = cv2.resize(padded[top:top+side, left:left+side], (128, 128))
        tensor = ((crop[:, :, ::-1].astype(np.float32) / 127.5) - 1).transpose(2, 0, 1)[None]
        start = time.perf_counter()
        result = self.stargan.run(None, {'images': tensor, 'attributes': np.asarray([targets], np.float32)})[0]
        milliseconds = (time.perf_counter() - start) * 1000
        if result.shape != (1, 3, 128, 128) or not np.isfinite(result).all():
            raise ValueError("模型输出形状或数值无效")
        edited = np.uint8(np.rint(np.clip((result[0].transpose(1, 2, 0) + 1) * 127.5, 0, 255)))[:, :, ::-1]
        return {'input_crop': encode_image(crop), 'image': encode_image(edited),
                'model_ms': round(milliseconds, 3), 'targets': targets,
                'note': 'StarGAN合成结果，128×128人脸裁剪；目标标签不是对人物真实属性的判断。'}

    def process(self, frame: np.ndarray, effect: str, strength: float, landmarks: bool) -> tuple:
        if effect not in EFFECTS:
            raise ValueError("未知特效")
        start = time.perf_counter()
        faces = self.detect(frame)
        result = render_effect(frame, faces, effect, strength, landmarks)
        elapsed = (time.perf_counter() - start) * 1000
        return result, {"faces": len(faces), "pipeline_ms": round(elapsed, 2),
                        "pipeline_fps": round(1000 / max(elapsed, .001), 1),
                        "width": frame.shape[1], "height": frame.shape[0],
                        "landmarks": [face[4:14].reshape(5, 2).tolist() for face in faces]}


def render_effect(frame: np.ndarray, faces: np.ndarray, effect: str = "none",
                  strength: float = .5, landmarks: bool = True) -> np.ndarray:
    """Five-point geometric effects; makeup is approximate, not semantic segmentation."""
    strength = float(np.clip(strength, 0, 1))
    out = frame.copy()
    for face in faces:
        x, y, w, h = face[:4]
        pts = face[4:14].reshape(5, 2)
        eyes = sorted(pts[:2], key=lambda p: p[0])
        left, right = np.asarray(eyes[0]), np.asarray(eyes[1])
        delta = right - left
        distance = max(float(np.linalg.norm(delta)), 1)
        ux = delta / distance
        uy = np.array([-ux[1], ux[0]])
        mid = (left + right) / 2

        def p(a: float, b: float) -> tuple[int, int]:
            return tuple(np.rint(mid + distance * (a * ux + b * uy)).astype(int))

        if effect in {"beauty", "all"}:
            mask = np.zeros(frame.shape[:2], np.uint8)
            cv2.ellipse(mask, (int(x + w / 2), int(y + h / 2)),
                        (max(1, int(w * .43)), max(1, int(h * .48))), 0, 0, 360, 255, -1)
            alpha = cv2.GaussianBlur(mask, (31, 31), 0)[..., None] / 255 * strength * .8
            softened = cv2.bilateralFilter(out, 7, 45, 45).astype(float)
            softened = np.clip(softened * 1.04 + 3, 0, 255)
            out = np.uint8(out * (1 - alpha) + softened * alpha)
        if effect in {"glasses", "all"}:
            layer = out.copy()
            for offset in (-.5, .5):
                poly = np.array([p(offset-.34, -.24), p(offset+.34, -.24),
                                 p(offset+.31, .23), p(offset-.31, .23)])
                cv2.fillConvexPoly(layer, poly, (35, 24, 33))
                cv2.polylines(layer, [poly], True, (220, 190, 105), max(2, int(distance*.05)))
            cv2.line(layer, p(-.16, -.04), p(.16, -.04), (220, 190, 105), max(2, int(distance*.05)))
            out = cv2.addWeighted(layer, .9, out, .1, 0)
        if effect in {"crown", "all"}:
            poly = np.array([p(-.9, -.6), p(-1, -1.35), p(-.45, -.97),
                             p(0, -1.65), p(.45, -.97), p(1, -1.35), p(.9, -.6)])
            cv2.fillPoly(out, [poly], (66, 177, 245))
            cv2.polylines(out, [poly], True, (160, 229, 255), max(2, int(distance*.03)))
        if effect in {"lipstick", "all"}:
            mouth = pts[3:5]
            center = tuple(np.rint(mouth.mean(axis=0)).astype(int))
            mw = max(2, int(np.linalg.norm(mouth[1] - mouth[0]) / 2))
            angle = float(np.degrees(np.arctan2(delta[1], delta[0])))
            mask = np.zeros(frame.shape[:2], np.uint8)
            # ponytail: five-point ellipse is approximate; use dense lip segmentation for production.
            cv2.ellipse(mask, center, (mw, max(2, int(mw*.22))), angle, 0, 360, 255, -1)
            alpha = cv2.GaussianBlur(mask, (5, 5), 0)[..., None] / 255 * strength * .7
            out = np.uint8(out * (1-alpha) + np.array([75, 40, 205]) * alpha)
        if landmarks:
            cv2.rectangle(out, (int(x), int(y)), (int(x+w), int(y+h)), (110, 235, 175), 1)
            for point in pts:
                cv2.circle(out, tuple(np.rint(point).astype(int)), 3, (75, 240, 175), -1)
    return out


class Handler(BaseHTTPRequestHandler):
    vision: Vision

    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def log_message(self, format, *args):
        pass  # Image contents and request bodies are never logged.

    def reply(self, status: int, data: bytes, mime: str):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def json_reply(self, status: int, value: dict):
        self.reply(status, json.dumps(value, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def allowed(self) -> bool:
        hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        return host in hosts and (origin is None or origin in {f"http://{h}" for h in hosts})

    def do_GET(self):
        if not self.allowed():
            return self.json_reply(403, {"error": "仅允许本机访问"})
        path = urlsplit(self.path).path
        if path == "/health":
            return self.json_reply(200, {"status": "ok", "backend": "OpenCV YuNet + SFace", "device": "CPU",
                                         "stargan_available": STARGAN_MODEL.is_file()})
        allowed_files = {"/": (ROOT / "web/index.html", "text/html; charset=utf-8"),
                         "/sample.jpg": (ROOT / "assets/sample.jpg", "image/jpeg"),
                         "/tutorial": (ROOT / "docs/tutorial.html", "text/html; charset=utf-8"),
                         "/tutorial.html": (ROOT / "docs/tutorial.html", "text/html; charset=utf-8"),
                         "/3d": (ROOT / "reports/3d/viewer.html", "text/html; charset=utf-8"),
                         "/face.obj": (ROOT / "reports/3d/face.obj", "text/plain; charset=utf-8"),
                         "/multiview.png": (ROOT / "reports/3d/multiview.png", "image/png")}
        for name in ("code-compendium", "code-map", "issues-and-fixes", "project-report",
                     "requirements-matrix", "sources", "presentation-outline", "tutorial-zh", "continued-experiments", "desktop-guide"):
            allowed_files[f"/{name}.md"] = (ROOT / f"docs/{name}.md", "text/plain; charset=utf-8")
            allowed_files[f"/{name}.html"] = (ROOT / f"docs/{'tutorial' if name == 'tutorial-zh' else name}.html", "text/html; charset=utf-8")
        if path not in allowed_files:
            return self.json_reply(404, {"error": "页面不存在"})
        file, mime = allowed_files[path]
        if not file.exists():
            return self.json_reply(404, {"error": "此文件尚未生成"})
        self.reply(200, file.read_bytes(), mime)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 22_000_000:
                raise ValueError("请求大小无效")
            # Consume a bounded body before rejecting, avoiding Windows TCP resets.
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("请求内容不完整")
            if not self.allowed():
                return self.json_reply(403, {"error": "仅允许本机访问"})
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                raise ValueError("仅接受 JSON 请求")
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("请求必须是对象")
            if self.path == "/process":
                strength = float(data.get("strength", .5))
                if not np.isfinite(strength) or not 0 <= strength <= 1:
                    raise ValueError("强度必须在 0–1 之间")
                frame, metrics = self.vision.process(decode_image(data.get("image")),
                    data.get("effect", "none"), strength, bool(data.get("landmarks", True)))
                return self.json_reply(200, {"image": encode_image(frame), **metrics})
            if self.path == "/verify":
                threshold = float(data.get("threshold", .363))
                if not np.isfinite(threshold) or not -1 <= threshold <= 1:
                    raise ValueError("阈值必须在 -1–1 之间")
                return self.json_reply(200, self.vision.verify(decode_image(data.get("first")),
                    decode_image(data.get("second")), threshold))
            if self.path == "/attributes":
                return self.json_reply(200, self.vision.edit_attributes(decode_image(data.get("image")), data.get("targets")))
            return self.json_reply(404, {"error": "接口不存在"})
        except (ValueError, TypeError, cv2.error) as exc:
            self.json_reply(400, {"error": str(exc)})
        except TimeoutError:
            self.json_reply(408, {"error": "请求超时"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", choices=["127.0.0.1", "0.0.0.0"], default="127.0.0.1",
                        help="Use 0.0.0.0 only inside a container mapped to host loopback")
    args = parser.parse_args()
    Handler.vision = Vision()
    # ponytail: one CPU request at a time keeps mutable OpenCV model state safe.
    server = HTTPServer((args.host, args.port), Handler)
    server.timeout = 30
    print(f"Face Vision Lab: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
