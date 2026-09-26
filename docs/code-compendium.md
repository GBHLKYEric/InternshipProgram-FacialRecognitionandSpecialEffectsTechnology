# 全部项目源码与使用位置

此文档逐字收录本项目源文件，并提供使用位置、符号行号、原文件链接与SHA256。它由 `scripts/build_deliverables.py` 从实际文件自动生成，避免手工粘贴漏代码。第三方Python库与预训练权重不复制进本文；它们的版本、官方来源与许可证见环境锁定文件、`models/registry.json`、`models/licenses/`及`docs/sources.md`。

## 如何阅读
先查用途，再打开对应文件。文中的代码是完整源码，不是删节片段。Notebook收录全部代码单元，图像输出在原始ipynb中。算法的专业解释见主教程。

## 文件目录

| 文件 | 源码行数 | 使用位置 |
|---|---:|---|
| `.dockerignore` | 9 | 依赖固定、打包或版本控制配置；由 pip、Docker 或 Git 读取。 |
| `.gitignore` | 39 | 依赖固定、打包或版本控制配置；由 pip、Docker 或 Git 读取。 |
| `app.py` | 250 | 本地网页服务；web/index.html 调用 /process 和 /verify。PDF任务2.4、4.3、9.2、9.3及系统集成。 |
| `Dockerfile` | 16 | docker build --target hello 或 --target lab；PDF任务1.3。 |
| `hello_world.py` | 4 | Docker hello镜像启动命令；PDF任务1.2/1.3。保留了旧仓库hello.py目录。 |
| `notebooks/01-first-vision-lab.ipynb` | 240 | Jupyter中的循序实验；PDF任务1.4、2.1、2.4和OpenCV入门。 |
| `requirements-docs.txt` | 7 | 依赖固定、打包或版本控制配置；由 pip、Docker 或 Git 读取。 |
| `requirements-research.txt` | 8 | 依赖固定、打包或版本控制配置；由 pip、Docker 或 Git 读取。 |
| `requirements.txt` | 2 | 依赖固定、打包或版本控制配置；由 pip、Docker 或 Git 读取。 |
| `research/__init__.py` | 1 | 研究训练与评估命令使用；详见 research/README.md 和 docs/code-map.md。对应PDF任务3.2–7.3。 |
| `research/configs/wider_retinanet.py` | 19 | 研究训练与评估命令使用；详见 research/README.md 和 docs/code-map.md。对应PDF任务3.2–7.3。 |
| `research/data.py` | 58 | 研究训练与评估命令使用；详见 research/README.md 和 docs/code-map.md。对应PDF任务3.2–7.3。 |
| `research/fetch_wider_subset.py` | 132 | 研究训练与评估命令使用；详见 research/README.md 和 docs/code-map.md。对应PDF任务3.2–7.3。 |
| `research/landmarks.py` | 205 | 研究训练与评估命令使用；详见 research/README.md 和 docs/code-map.md。对应PDF任务3.2–7.3。 |
| `research/lfw.py` | 202 | 研究训练与评估命令使用；详见 research/README.md 和 docs/code-map.md。对应PDF任务3.2–7.3。 |
| `research/optimize.py` | 110 | 研究训练与评估命令使用；详见 research/README.md 和 docs/code-map.md。对应PDF任务3.2–7.3。 |
| `research/prepare_lfw_pilot.py` | 66 | 研究训练与评估命令使用；详见 research/README.md 和 docs/code-map.md。对应PDF任务3.2–7.3。 |
| `research/recognition.py` | 203 | 研究训练与评估命令使用；详见 research/README.md 和 docs/code-map.md。对应PDF任务3.2–7.3。 |
| `research/smoke.py` | 91 | 研究训练与评估命令使用；详见 research/README.md 和 docs/code-map.md。对应PDF任务3.2–7.3。 |
| `research/stargan.py` | 197 | 研究训练与评估命令使用；详见 research/README.md 和 docs/code-map.md。对应PDF任务3.2–7.3。 |
| `scripts/build_deliverables.py` | 214 | 模型下载、环境检查、数据准备或交付物生成；本文件开头docstring和下方函数索引提供具体入口。 |
| `scripts/demo.py` | 68 | 模型下载、环境检查、数据准备或交付物生成；本文件开头docstring和下方函数索引提供具体入口。 |
| `scripts/doctor.py` | 60 | 模型下载、环境检查、数据准备或交付物生成；本文件开头docstring和下方函数索引提供具体入口。 |
| `scripts/fetch_models.py` | 41 | 模型下载、环境检查、数据准备或交付物生成；本文件开头docstring和下方函数索引提供具体入口。 |
| `scripts/make_notebook.py` | 25 | 模型下载、环境检查、数据准备或交付物生成；本文件开头docstring和下方函数索引提供具体入口。 |
| `scripts/mmdet.Dockerfile` | 9 | 模型下载、环境检查、数据准备或交付物生成；本文件开头docstring和下方函数索引提供具体入口。 |
| `scripts/mmdet_pilot.py` | 91 | 模型下载、环境检查、数据准备或交付物生成；本文件开头docstring和下方函数索引提供具体入口。 |
| `scripts/verify_local.py` | 42 | 模型下载、环境检查、数据准备或交付物生成；本文件开头docstring和下方函数索引提供具体入口。 |
| `start.ps1` | 17 | Windows本地启动与环境准备入口；项目根执行 powershell -File start.ps1。 |
| `tests/test_app.py` | 42 | 回归检查；在项目根目录运行 python -m unittest discover -s tests。 |
| `tests/test_research.py` | 39 | 回归检查；在项目根目录运行 python -m unittest discover -s tests。 |
| `vision3d/__init__.py` | 1 | 由 python -m vision3d.reconstruct 及其导出脚本调用；PDF任务8.2、8.3。 |
| `vision3d/check.py` | 37 | 由 python -m vision3d.reconstruct 及其导出脚本调用；PDF任务8.2、8.3。 |
| `vision3d/mobilenet_v1.py` | 163 | 由 python -m vision3d.reconstruct 及其导出脚本调用；PDF任务8.2、8.3。 |
| `vision3d/reconstruct.py` | 183 | 由 python -m vision3d.reconstruct 及其导出脚本调用；PDF任务8.2、8.3。 |
| `vision3d/viewer-template.html` | 20 | 由 python -m vision3d.reconstruct 及其导出脚本调用；PDF任务8.2、8.3。 |
| `web/index.html` | 49 | 由 app.py 的 GET / 返回；浏览器处理图像输入、效果控制、摄像头帧和双图验证。 |

共 37 个源文件，2960 行文本（Notebook按JSON行统计）。


## .dockerignore

**使用位置：** 依赖固定、打包或版本控制配置；由 pip、Docker 或 Git 读取。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/.dockerignore)

SHA256：`f007018a9f97243fceba0724ac2b6ce9fcf272a6a462856d48d7134688f9c31b`

````text
.venv
.git
data
runs
reports
*.zip
*.pdf
*.pptx
__pycache__

````


## .gitignore

**使用位置：** 依赖固定、打包或版本控制配置；由 pip、Docker 或 Git 读取。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/.gitignore)

SHA256：`cebaabc398a2626e8ff12f4b1b0cc564181df03c6c47fa82305ad9f5dca48493`

````text
.venv/
__pycache__/
*.pyc
.pytest_cache/
models/*.onnx
models/*.pth
models/*.pt
data/
runs/
*.log
.env
*.mp4
*.avi
*.zip
*.pdf
*.pptx
assets/sample.jpg
artifacts/*.onnx
artifacts/*.pt
artifacts/*.pth
artifacts/*.obj
artifacts/*.mp4
models/3ddfa/
models/*.npz
reports/*.mp4
reports/*.obj
reports/*.ply
reports/**/*.onnx
reports/**/*.pt
reports/**/*.pth
reports/**/*.obj
reports/**/*.ply
reports/**/*.npz
reports/**/*.jpg
reports/**/*.png
reports/*.jpg
reports/*.png
reports/3d/viewer.html
assets/*.png

````


## app.py

**使用位置：** 本地网页服务；web/index.html 调用 /process 和 /verify。PDF任务2.4、4.3、9.2、9.3及系统集成。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/app.py)

SHA256：`c6cc5f2e885313173dbf3f9a88b68e87421ac82bdf991f1de1c3dbeba8869939`

**代码定位：** `decode_image` 第19行；`encode_image` 第37行；`Vision` 第44行；`render_effect` 第85行；`Handler` 第143行；`main` 第230行。

````python
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
            return self.json_reply(200, {"status": "ok", "backend": "OpenCV YuNet + SFace", "device": "CPU"})
        allowed_files = {"/": (ROOT / "web/index.html", "text/html; charset=utf-8"),
                         "/sample.jpg": (ROOT / "assets/sample.jpg", "image/jpeg"),
                         "/tutorial": (ROOT / "docs/tutorial.html", "text/html; charset=utf-8"),
                         "/3d": (ROOT / "reports/3d/viewer.html", "text/html; charset=utf-8"),
                         "/face.obj": (ROOT / "reports/3d/face.obj", "text/plain; charset=utf-8"),
                         "/multiview.png": (ROOT / "reports/3d/multiview.png", "image/png")}
        for name in ("code-compendium", "code-map", "issues-and-fixes", "project-report",
                     "requirements-matrix", "sources", "presentation-outline", "tutorial-zh"):
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

````


## Dockerfile

**使用位置：** docker build --target hello 或 --target lab；PDF任务1.3。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/Dockerfile)

SHA256：`ec1fd3c97493505d52a9256b987ef2ead772c79d7523779eee264b2fbc699996`

````text
FROM python:3.12-slim AS hello
WORKDIR /app
COPY hello_world.py .
CMD ["python", "hello_world.py"]

FROM hello AS lab
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY scripts scripts
COPY web web
COPY docs docs
COPY models models
RUN python scripts/fetch_models.py
CMD ["python", "app.py", "--host", "0.0.0.0"]

````


## hello_world.py

**使用位置：** Docker hello镜像启动命令；PDF任务1.2/1.3。保留了旧仓库hello.py目录。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/hello_world.py)

SHA256：`4e3bb33bd21d7fe65c477e05a35ca5df27ab3086f5133af0d086c09c33425680`

````python
"""PDF task 1.2/1.3: first reproducible container program."""
import platform

print(f"Hello, Face Vision Lab! Python {platform.python_version()}")

````


## notebooks/01-first-vision-lab.ipynb

**使用位置：** Jupyter中的循序实验；PDF任务1.4、2.1、2.4和OpenCV入门。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/notebooks/01-first-vision-lab.ipynb)

SHA256：`09dcf34b45466cc4c5b0f21c359735edb25de756c3360c03bf23f7c55309040a`

### 第 2 个单元

```python
from pathlib import Path
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
ROOT = Path.cwd()
if ROOT.name == 'notebooks': ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from app import Vision
print('Python:', sys.version.split()[0], 'OpenCV:', cv2.__version__)
```

### 第 4 个单元

```python
image = cv2.imread(str(ROOT / 'assets/sample.jpg'))
assert image is not None
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print('彩色形状:', image.shape, '灰度形状:', gray.shape)
fig, ax = plt.subplots(1, 2, figsize=(8,4))
ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)); ax[1].imshow(gray, cmap='gray')
for a in ax: a.axis('off')
plt.show()
```

### 第 6 个单元

```python
vision = Vision(ROOT / 'models')
faces = vision.detect(image)
print('检测到的人脸:', len(faces))
print('眼睛、鼻尖、嘴角:', faces[0,4:14].reshape(5,2))
result, metrics = vision.process(image, 'all', .6, True)
plt.figure(figsize=(5,5)); plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)); plt.axis('off'); plt.show()
print(metrics)
```

### 第 8 个单元

```python
def cosine(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
print(cosine([1,0],[1,0]), cosine([1,0],[0,1]), cosine([1,0],[-1,0]))
assert cosine([1,0],[0,1]) == 0
print(vision.verify(image,image,.363))
```


## requirements-docs.txt

**使用位置：** 依赖固定、打包或版本控制配置；由 pip、Docker 或 Git 读取。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/requirements-docs.txt)

SHA256：`dd3db4f423fa2a389570f45a47dc9922b6c58a288f3763f9e307af62272832ea`

````text
markdown==3.11
beautifulsoup4==4.15.0
python-pptx==1.0.2
reportlab==5.0.1
nbformat==5.11.1
nbclient==0.11.0
jupyterlab==4.6.4

````


## requirements-research.txt

**使用位置：** 依赖固定、打包或版本控制配置；由 pip、Docker 或 Git 读取。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/requirements-research.txt)

SHA256：`976c19ef8085b31b1a846c4bc1aaf019399c2298325835903f7741243590cdf5`

````text
-r requirements.txt
torch==2.14.0
torchvision==0.29.0
onnx==1.23.0
onnxruntime==1.30.0
scipy==1.18.1
matplotlib==3.11.2
Pillow==12.3.0

````


## requirements.txt

**使用位置：** 依赖固定、打包或版本控制配置；由 pip、Docker 或 Git 读取。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/requirements.txt)

SHA256：`00a92a9992981d08dff3533e78748e2304cf7fcdca91d8fa7ed5f9ac00cff1e2`

````text
numpy==2.5.3
opencv-python==5.0.0.93

````


## research/__init__.py

**使用位置：** 研究训练与评估命令使用；详见 research/README.md 和 docs/code-map.md。对应PDF任务3.2–7.3。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/research/__init__.py)

SHA256：`3f73ddc5e4fd7ff98d83c64b00895186989d2106c07a4129efd9580691d8d526`

````python
"""Explicit research pipelines. No benchmark scores are invented by this package."""

````


## research/configs/wider_retinanet.py

**使用位置：** 研究训练与评估命令使用；详见 research/README.md 和 docs/code-map.md。对应PDF任务3.2–7.3。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/research/configs/wider_retinanet.py)

SHA256：`be96a3fcd877a378b901d6da96b1b51ee0360dbe2d203b37d5a74cac1cf6d307`

````python
"""MMDetection 3.3 / MMCV 2.1 / MMEngine 0.10; use its separate Linux environment.

RetinaNet is a practical detector baseline, not the RetinaFace algorithm.
Paths are relative to the project root. COCO AP is not WIDER's official subset AP.
"""
_base_ = 'mmdet::retinanet/retinanet_r50_fpn_1x_coco.py'
model = dict(bbox_head=dict(num_classes=1))
metainfo = dict(classes=('face',), palette=[(255, 100, 100)])
train_dataloader = dict(batch_size=2, num_workers=2, dataset=dict(
    data_root='data/', ann_file='wider/train.json', data_prefix=dict(img='WIDER_train/images/'), metainfo=metainfo))
val_dataloader = dict(batch_size=1, num_workers=2, dataset=dict(
    data_root='data/', ann_file='wider/val.json', data_prefix=dict(img='WIDER_val/images/'), metainfo=metainfo))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file='data/wider/val.json', metric='bbox', classwise=True)
test_evaluator = val_evaluator
train_cfg = dict(max_epochs=12)
optim_wrapper = dict(optimizer=dict(lr=.00125))
default_hooks = dict(checkpoint=dict(interval=1, max_keep_ckpts=2))
work_dir = 'runs/wider_retinanet'

````


## research/data.py

**使用位置：** 研究训练与评估命令使用；详见 research/README.md 和 docs/code-map.md。对应PDF任务3.2–7.3。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/research/data.py)

SHA256：`8edea6a8e16437d43b9e5053e9e8af588205cab39e3441c43a4e5b33a845207f`

**代码定位：** `wider_to_coco` 第12行。

````python
"""Convert WIDER FACE annotations to a validated single-class COCO dataset.

python -m research.data --images data/WIDER_train/images --annotations data/wider_face_train_bbx_gt.txt --output data/wider/train.json
Invalid WIDER boxes are excluded. COCO mAP is not the official WIDER Easy/Medium/Hard AP.
"""
import argparse
import json
from pathlib import Path
from PIL import Image


def wider_to_coco(images, annotations, output):
    root = Path(images).resolve()
    lines = Path(annotations).read_text(encoding='utf-8').strip().splitlines()
    result = {'images':[],'annotations':[],'categories':[{'id':1,'name':'face'}]}
    cursor = 0
    excluded = 0
    seen = set()
    while cursor < len(lines):
        name = lines[cursor].strip(); cursor += 1
        if cursor >= len(lines): raise ValueError('Missing box count')
        count = int(lines[cursor]); cursor += 1
        if count < 0: raise ValueError('Negative box count')
        path = (root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or name in seen:
            raise ValueError(f'Missing, duplicate, or unsafe WIDER image: {name}')
        seen.add(name)
        with Image.open(path) as image: width,height = image.size
        image_id = len(result['images'])+1
        result['images'].append({'id':image_id,'file_name':name,'width':width,'height':height})
        # Some official files include one all-zero sentinel after zero faces.
        if count == 0 and cursor < len(lines):
            fields = lines[cursor].split()
            if len(fields)==10 and all(value=='0' for value in fields): cursor+=1
        for _ in range(count):
            if cursor >= len(lines): raise ValueError('Truncated WIDER box annotations')
            fields = list(map(int,lines[cursor].split())); cursor+=1
            if len(fields)!=10: raise ValueError('Expected ten WIDER box annotation fields')
            x,y,w,h,blur,expression,illumination,invalid,occlusion,pose = fields
            if invalid or w<=0 or h<=0:
                excluded+=1; continue
            x1,y1=max(0,x),max(0,y); x2,y2=min(width,x+w),min(height,y+h)
            if x2<=x1 or y2<=y1:
                excluded+=1; continue
            result['annotations'].append({'id':len(result['annotations'])+1,'image_id':image_id,
                'category_id':1,'bbox':[x1,y1,x2-x1,y2-y1],'area':(x2-x1)*(y2-y1),'iscrowd':0})
    if not result['images']: raise ValueError('Empty WIDER annotation file')
    destination=Path(output); destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result),encoding='utf-8')
    report={'images':len(result['images']),'valid_boxes':len(result['annotations']),'excluded_boxes':excluded}
    print(json.dumps(report)); return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images',required=True); parser.add_argument('--annotations',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args(); wider_to_coco(args.images,args.annotations,args.output)

````


## research/fetch_wider_subset.py

**使用位置：** 研究训练与评估命令使用；详见 research/README.md 和 docs/code-map.md。对应PDF任务3.2–7.3。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/research/fetch_wider_subset.py)

SHA256：`4ec59a6008640b6d2467ec8e651369f60e36659e2e9802a33fc61617648d01e4`

**代码定位：** `fetch` 第22行；`RemoteZipFile` 第35行；`annotation_blocks` 第65行；`run` 第80行。

````python
"""Download a reproducible real WIDER FACE pilot subset via HTTP ZIP ranges.

python -m research.fetch_wider_subset --train 32 --val 16 --seed 42
The dataset remains under its original CC BY-NC-ND-4.0 terms, not this code's
license. Data stay local. Small pilot COCO AP is NOT official WIDER benchmark AP.
"""
import argparse
import hashlib
import io
import json
import random
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from research.data import wider_to_coco

REPOSITORY='CUHK-CSE/wider_face'


def fetch(url,headers=None,limit=32*1024*1024):
    for attempt in range(3):
        try:
            request=urllib.request.Request(url,headers={'Accept-Encoding':'identity',**(headers or {})})
            with urllib.request.urlopen(request,timeout=60) as response:
                data=response.read(limit+1)
                if len(data)>limit: raise ValueError('Response exceeds bounded download size')
                return data,response.status,response.headers
        except (urllib.error.URLError,TimeoutError):
            if attempt==2: raise
            time.sleep(attempt+1)


class RemoteZipFile(io.RawIOBase):
    """Minimal seekable HTTP file for ZipFile; rejects servers ignoring Range."""
    def __init__(self,url,size):
        self.url,self.size,self.position=url,size,0
        self.cache_start=0; self.cache=b''; self.downloaded=0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.position

    def seek(self,offset,whence=0):
        position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if whence not in {0,1,2} or not 0<=position<=self.size: raise ValueError('Invalid remote seek')
        self.position=position; return position

    def read(self,size=-1):
        size=self.size-self.position if size<0 else min(size,self.size-self.position)
        if size==0: return b''
        if size>32*1024*1024: raise ValueError('ZIP member exceeds 32 MiB range-read guard')
        if not self.cache_start<=self.position or self.position+size>self.cache_start+len(self.cache):
            end=min(self.size,self.position+max(size,256*1024))-1
            expected=f'bytes {self.position}-{end}/{self.size}'
            data,status,headers=fetch(self.url,{'Range':f'bytes={self.position}-{end}'},end-self.position+1)
            if status!=206 or headers.get('Content-Range')!=expected or len(data)!=end-self.position+1:
                raise RuntimeError('Server ignored/mismatched HTTP Range; refusing a silent full-archive download')
            self.cache_start,self.cache=self.position,data; self.downloaded+=len(data)
        start=self.position-self.cache_start; self.position+=size
        return self.cache[start:start+size]


def annotation_blocks(text):
    lines=text.strip().splitlines(); cursor=0; result={}
    while cursor<len(lines):
        start=cursor; name=lines[cursor].strip(); cursor+=1
        if name in result or cursor>=len(lines): raise ValueError('Duplicate/truncated official annotations')
        count=int(lines[cursor]); cursor+=1
        if count<0 or cursor+count>len(lines): raise ValueError('Invalid official box count')
        cursor+=count
        if count==0 and cursor<len(lines):
            fields=lines[cursor].split()
            if len(fields)==10 and all(value=='0' for value in fields): cursor+=1
        result[name]='\n'.join(lines[start:cursor])+'\n'
    return result


def run(args):
    if min(args.train,args.val)<1: raise ValueError('Both train and val subset counts must be positive')
    root=Path(args.output).resolve(); root.mkdir(parents=True,exist_ok=True)
    metadata=json.loads(fetch(f'https://huggingface.co/api/datasets/{REPOSITORY}')[0])
    revision=metadata['sha']
    tree=json.loads(fetch(f'https://huggingface.co/api/datasets/{REPOSITORY}/tree/{revision}/data')[0])
    entries={Path(entry['path']).name:entry for entry in tree if entry['type']=='file'}
    base=f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{revision}/data'
    annotation_bytes,_,_=fetch(base+'/wider_face_split.zip')
    expected=entries['wider_face_split.zip']['lfs']['oid']
    if hashlib.sha256(annotation_bytes).hexdigest()!=expected: raise ValueError('Official annotation archive SHA256 mismatch')
    report={'dataset':REPOSITORY,'revision':revision,'license':'CC-BY-NC-ND-4.0',
        'purpose':'Real tiny training/validation pilot; not a standard benchmark or synthetic fixture',
        'selection':'Uniform sample without replacement from sorted annotated image names, independently within official train/val splits',
        'seed':args.seed,'annotations_sha256_verified':expected,'splits':{}}
    with zipfile.ZipFile(io.BytesIO(annotation_bytes)) as annotations:
        for split,count in [('train',args.train),('val',args.val)]:
            member=next(name for name in annotations.namelist() if name.endswith(f'wider_face_{split}_bbx_gt.txt'))
            blocks=annotation_blocks(annotations.read(member).decode('utf-8'))
            if count>len(blocks): raise ValueError('Requested more examples than split contains')
            selected=sorted(random.Random(args.seed).sample(sorted(blocks),count))
            archive_name=f'WIDER_{split}.zip'; entry=entries[archive_name]
            remote=RemoteZipFile(base+'/'+archive_name,entry['size'])
            image_root=root/f'WIDER_{split}'/'images'; image_root.mkdir(parents=True,exist_ok=True)
            hashes={}
            with zipfile.ZipFile(remote) as archive:
                for index,name in enumerate(selected):
                    destination=(image_root/name).resolve()
                    if not destination.is_relative_to(image_root): raise ValueError('Unsafe image archive path')
                    zip_name=f'WIDER_{split}/images/{name}'
                    if archive.getinfo(zip_name).file_size>32*1024*1024: raise ValueError('Uncompressed image exceeds size guard')
                    content=archive.read(zip_name)  # zipfile verifies each member's CRC32.
                    destination.parent.mkdir(parents=True,exist_ok=True); destination.write_bytes(content)
                    hashes[name]=hashlib.sha256(content).hexdigest()
                    print(f'WIDER {split}: {index+1}/{count}',flush=True)
            annotation_path=root/'wider'/f'subset_{split}_bbx_gt.txt'; annotation_path.parent.mkdir(parents=True,exist_ok=True)
            annotation_path.write_text(''.join(blocks[name] for name in selected),encoding='utf-8')
            conversion=wider_to_coco(image_root,annotation_path,root/'wider'/f'{split}.json')
            report['splits'][split]={**conversion,'image_sha256':hashes,'http_bytes_downloaded':remote.downloaded,
                'source_archive_sha256_advertised_not_fully_verified':entry['lfs']['oid'],
                'integrity':'Pinned repository revision; each downloaded ZIP member CRC32 verified; local images individually SHA256 recorded'}
    destination=Path(args.report); destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({name:{k:v for k,v in value.items() if k!='image_sha256'} for name,value in report['splits'].items()},indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',type=int,default=32); parser.add_argument('--val',type=int,default=16)
    parser.add_argument('--seed',type=int,default=42); parser.add_argument('--output',default='data')
    parser.add_argument('--report',default='reports/wider-subset.json')
    run(parser.parse_args())

````


## research/landmarks.py

**使用位置：** 研究训练与评估命令使用；详见 research/README.md 和 docs/code-map.md。对应PDF任务3.2–7.3。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/research/landmarks.py)

SHA256：`5a2718bf1a6cfa1917c5269ab07034984368ce539446d0f206818657c6a21161`

**代码定位：** `nme` 第13行；`prepare` 第25行；`dataset` 第54行；`make_model` 第96行；`train` 第104行；`align` 第146行；`infer` 第161行。

````python
"""300W 68-point coordinate-regression baseline, NME and affine alignment.

prepare: convert original .pts files to JSON manifest; manually provide disjoint
train/validation roots. This small ResNet18 baseline is not HRNet or SAN.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def nme(prediction, target):
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape or target.ndim != 3 or target.shape[1:] != (68,2):
        raise ValueError('Expected equal (N,68,2) arrays in original pixel coordinates')
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError('Landmarks must be finite')
    distance = np.linalg.norm(target[:,36]-target[:,45], axis=1)
    if (distance<=0).any():
        raise ValueError('Outer eye-corner distance must be positive')
    return np.linalg.norm(prediction-target, axis=2).mean(1)/distance


def prepare(root, output):
    from PIL import Image
    root = Path(root).resolve()
    entries = []
    for pts in sorted(root.rglob('*.pts')):
        text = pts.read_text(encoding='utf-8')
        if '{' not in text or '}' not in text:
            raise ValueError(f'Invalid 300W PTS: {pts}')
        coordinates = np.array([list(map(float,line.split())) for line in text.split('{',1)[1].split('}',1)[0].strip().splitlines()])
        if coordinates.shape != (68,2) or not np.isfinite(coordinates).all():
            raise ValueError(f'Expected 68 finite points: {pts}')
        matches = [pts.with_suffix(extension) for extension in ['.jpg','.png','.jpeg'] if pts.with_suffix(extension).is_file()]
        if len(matches) != 1:
            raise ValueError(f'Need exactly one image for {pts}')
        with Image.open(matches[0]) as image:
            width,height = image.size
        # Original 300W .pts coordinates are one-based; OpenCV/PIL use zero-based.
        coordinates -= 1
        left,top = np.maximum(coordinates.min(0)-.2*np.ptp(coordinates, axis=0), 0)
        right,bottom = np.minimum(coordinates.max(0)+.2*np.ptp(coordinates, axis=0), [width,height])
        entries.append({'image':str(matches[0].relative_to(root)), 'points':coordinates.tolist(),
                        'box':[int(left), int(top), int(np.ceil(right)), int(np.ceil(bottom))]})
    if not entries:
        raise FileNotFoundError('No 300W .pts annotations found')
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(entries), encoding='utf-8')
    print(f'Prepared {len(entries)} original annotated images')


def dataset(root, manifest):
    import torch
    from PIL import Image
    from torch.utils.data import Dataset
    from research.recognition import face_transform
    root = Path(root).resolve()
    entries = json.loads(Path(manifest).read_text(encoding='utf-8'))
    if not isinstance(entries, list) or not entries:
        raise ValueError('Manifest must be a nonempty JSON list')
    for item in entries:
        path = (root/item['image']).resolve()
        points = np.asarray(item['points'], dtype=np.float32)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Unsafe or missing image: {path}')
        if points.shape != (68,2) or not np.isfinite(points).all():
            raise ValueError('Each entry needs 68 finite pixel coordinates')
        box = item['box']
        if len(box) != 4 or box[2]<=box[0] or box[3]<=box[1] or min(box)<0:
            raise ValueError('Invalid crop box')

    class FacePoints(Dataset):
        def __len__(self):
            return len(entries)

        def __getitem__(self, index):
            item = entries[index]
            with Image.open(root/item['image']) as image:
                image = image.convert('RGB')
                box = item['box']
                if box[2] > image.width or box[3] > image.height:
                    raise ValueError('Crop exceeds original image bounds')
                crop = image.crop(box)
                tensor = face_transform(size=128)(crop)
            origin = np.asarray(box[:2], dtype=np.float32)
            size = np.asarray(box[2:], dtype=np.float32)-origin
            points = np.asarray(item['points'], dtype=np.float32)
            return tensor, torch.from_numpy((points-origin)/size), torch.from_numpy(origin), torch.from_numpy(size)
    result = FacePoints()
    result.paths = {(root/item['image']).resolve() for item in entries}
    return result


def make_model():
    from torchvision.models import resnet18
    from torch import nn
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 136)
    return model


def train(args):
    import torch
    from torch.utils.data import DataLoader
    from research.recognition import seed_all
    seed_all(args.seed)
    training = dataset(args.root, args.train)
    validation = dataset(args.val_root or args.root, args.val)
    if training.paths & validation.paths:
        raise ValueError('Training and validation manifests overlap; split original images first')
    if args.batch_size<2 or args.epochs<1 or len(training)<2:
        raise ValueError('Need >=2 training images, batch_size>=2 and epochs>=1')
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, drop_last=True)
    if not len(train_loader):
        raise ValueError('batch_size exceeds training dataset size')
    val_loader = DataLoader(validation, batch_size=args.batch_size)
    model = make_model().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    with (output/'history.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file); writer.writerow(['epoch','train_mse','val_nme_outer_eye_corners'])
        for epoch in range(args.epochs):
            model.train(); losses=[]
            for images, target, _, _ in train_loader:
                optimizer.zero_grad(set_to_none=True)
                prediction = model(images.to(args.device)).reshape(-1,68,2)
                loss = (prediction-target.to(args.device)).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite landmark loss')
                loss.backward(); optimizer.step(); losses.append(loss.item())
            model.eval(); scores=[]
            with torch.inference_mode():
                for images,target,origin,size in val_loader:
                    prediction = model(images.to(args.device)).cpu().reshape(-1,68,2)
                    prediction = prediction*size[:,None,:]+origin[:,None,:]
                    target = target*size[:,None,:]+origin[:,None,:]
                    scores.extend(nme(prediction.numpy(),target.numpy()))
            row = [epoch+1,float(np.mean(losses)),float(np.mean(scores))]
            writer.writerow(row); file.flush(); print(row, flush=True)
            torch.save({'model_type':'resnet18_landmarks68','model':model.state_dict(),
                        'optimizer':optimizer.state_dict(),'epoch':epoch+1,'nme':row[2]},output/'last.pt')


def align(image, points, size=112):
    """68 landmarks -> five semantic points -> robust similarity affine warp."""
    import cv2
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (68,2) or not np.isfinite(points).all() or size<1:
        raise ValueError('Expected 68 finite points and positive output size')
    source = np.array([points[36:42].mean(0),points[42:48].mean(0),points[30],points[48],points[54]])
    reference = np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],
                          [41.5493,92.3655],[70.7299,92.2041]],np.float32)*(size/112)
    matrix, inliers = cv2.estimateAffinePartial2D(source,reference,method=cv2.LMEDS)
    if matrix is None or not np.isfinite(matrix).all():
        raise ValueError('Degenerate landmark geometry')
    return cv2.warpAffine(image,matrix,(size,size)), matrix


def infer(args):
    import cv2
    import torch
    from PIL import Image
    from research.recognition import face_transform
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if checkpoint.get('model_type')!='resnet18_landmarks68': raise ValueError('Expected trained 68-point landmark checkpoint')
    model=make_model().eval(); model.load_state_dict(checkpoint['model'])
    with Image.open(args.image) as image:
        image=image.convert('RGB'); width,height=image.size
        box=args.box or [0,0,width,height]
        if len(box)!=4 or min(box)<0 or box[2]<=box[0] or box[3]<=box[1] or box[2]>width or box[3]>height:
            raise ValueError('Box must be x1 y1 x2 y2 in original image bounds')
        tensor=face_transform(size=128)(image.crop(box))[None]
        frame=cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
    with torch.inference_mode(): points=model(tensor).reshape(68,2).numpy()
    points=points*np.array([box[2]-box[0],box[3]-box[1]])+np.array(box[:2])
    aligned,matrix=align(frame,points)
    for x,y in points: cv2.circle(frame,(int(round(x)),int(round(y))),2,(0,255,0),-1)
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    for name,canvas in [('landmarks.png',frame),('aligned.png',aligned)]:
        success,encoded=cv2.imencode('.png',canvas)
        if not success: raise RuntimeError('PNG encoding failed')
        encoded.tofile(output/name)
    (output/'points.json').write_text(json.dumps({'points':points.tolist(),'affine':matrix.tolist(),'box':box}),encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare_parser = commands.add_parser('prepare'); prepare_parser.add_argument('--root', required=True)
    prepare_parser.add_argument('--output', required=True)
    train_parser = commands.add_parser('train')
    for name in ['root','train','val']: train_parser.add_argument('--'+name,required=True)
    train_parser.add_argument('--val-root'); train_parser.add_argument('--output',default='runs/landmarks')
    train_parser.add_argument('--epochs',type=int,default=20); train_parser.add_argument('--batch-size',type=int,default=16)
    train_parser.add_argument('--lr',type=float,default=.001); train_parser.add_argument('--seed',type=int,default=42)
    train_parser.add_argument('--device',default='cpu')
    infer_parser=commands.add_parser('infer')
    for name in ['checkpoint','image']: infer_parser.add_argument('--'+name,required=True)
    infer_parser.add_argument('--box',nargs=4,type=int); infer_parser.add_argument('--output',default='runs/landmarks-inference')
    args=parser.parse_args()
    if args.command=='prepare': prepare(args.root,args.output)
    elif args.command=='train': train(args)
    else: infer(args)

````


## research/lfw.py

**使用位置：** 研究训练与评估命令使用；详见 research/README.md 和 docs/code-map.md。对应PDF任务3.2–7.3。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/research/lfw.py)

SHA256：`784febbe1d1e62a0dc00315f68f797bd618bcef3a63189315dc6a395d0ec9387`

**代码定位：** `parse_pairs` 第14行；`evaluate_scores` 第49行；`run` 第85行；`align_largest` 第126行；`run_sface` 第135行。

````python
"""Strict official LFW 6000-pair, ten-fold verification with held-out thresholds.

python -m research.lfw --root data/lfw --pairs data/pairs.txt --checkpoint runs/arcface/last.pt
Input images must follow the training alignment procedure. Identity-disjoint
training and benchmark preprocessing/provenance must be checked by the operator.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def parse_pairs(root, pairs_path, strict=True):
    root = Path(root).resolve()
    lines = Path(pairs_path).read_text(encoding='utf-8').strip().splitlines()
    try:
        folds, per_class = map(int, lines[0].split())
    except (ValueError, IndexError) as error:
        raise ValueError('Expected official pairs header: 10 300') from error
    if strict and (folds, per_class) != (10, 300):
        raise ValueError('Official evaluation requires header 10 300; no reduced benchmark')
    if folds < 2 or per_class < 1 or len(lines)-1 != folds*per_class*2:
        raise ValueError('Pair count does not match header; refusing partial evaluation')
    pairs = []
    for index, line in enumerate(lines[1:]):
        fields = line.split()
        expected_same = index % (per_class*2) < per_class
        if len(fields) != (3 if expected_same else 4):
            raise ValueError(f'Bad pair format or within-fold ordering at line {index+2}')
        name_a, number_a = fields[:2]
        name_b, number_b = (name_a, fields[2]) if expected_same else fields[2:]
        paths = []
        for name, number in [(name_a, number_a), (name_b, number_b)]:
            if '/' in name or '\\' in name or name in {'.','..'} or not number.isdecimal() or int(number) < 1:
                raise ValueError('Invalid identity or image index')
            path = (root/name/f'{name}_{int(number):04d}.jpg').resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise FileNotFoundError(f'Missing or unsafe LFW image: {path}')
            paths.append(path)
        if expected_same and paths[0] == paths[1]:
            raise ValueError('A positive pair cannot repeat the same image')
        if not expected_same and name_a == name_b:
            raise ValueError('A negative pair must have different identity names')
        pairs.append((*paths, expected_same, index//(per_class*2)))
    return pairs


def evaluate_scores(scores, labels, folds, far_target=.001, valid=None):
    scores, labels, folds = np.asarray(scores), np.asarray(labels, dtype=bool), np.asarray(folds)
    if scores.ndim != 1 or not (scores.shape == labels.shape == folds.shape) or not np.isfinite(scores).all():
        raise ValueError('Scores, labels and folds must be equal finite one-dimensional arrays')
    if not 0 <= far_target <= 1 or len(np.unique(folds)) < 2:
        raise ValueError('Need >=2 folds and FAR in [0,1]')
    valid = np.ones(scores.shape,dtype=bool) if valid is None else np.asarray(valid,dtype=bool)
    if valid.shape != scores.shape:
        raise ValueError('Pair validity mask must match scores')
    results = []
    for fold in np.unique(folds):
        train, test = folds != fold, folds == fold
        if any(not labels[mask].any() or labels[mask].all() for mask in [train, test, train & valid]):
            raise ValueError('Each training and evaluation split needs positive and negative pairs')
        # Candidate thresholds use training scores only. No test-set calibration.
        thresholds = np.r_[np.nextafter(scores[train & valid].min(), -np.inf),
                           np.nextafter(np.unique(scores[train & valid]), np.inf)]
        accuracy = np.array([np.mean(valid[train] & ((scores[train]>=threshold)==labels[train])) for threshold in thresholds])
        threshold = float(thresholds[int(accuracy.argmax())])
        negative = scores[train & ~labels & valid]
        far_values = np.array([np.sum(negative>=value)/np.sum(train & ~labels) for value in thresholds])
        far_threshold = float(thresholds[np.flatnonzero(far_values<=far_target)[0]])
        results.append({'fold':int(fold), 'accuracy':float(np.mean(valid[test] & ((scores[test]>=threshold)==labels[test]))),
                        'threshold':threshold, 'tar':float(np.mean(valid[test & labels] & (scores[test & labels]>=far_threshold))),
                        'far':float(np.mean(valid[test & ~labels] & (scores[test & ~labels]>=far_threshold))), 'far_threshold':far_threshold,
                        'test_pairs':int(test.sum()),'failed_pairs':int((test & ~valid).sum())})
    accuracies = [r['accuracy'] for r in results]
    return {'folds':results, 'accuracy_mean':float(np.mean(accuracies)),
            'accuracy_std':float(np.std(accuracies)), 'far_target':far_target,
            'tar_mean':float(np.mean([r['tar'] for r in results])),
            'far_mean':float(np.mean([r['far'] for r in results])),
            'failed_pairs':int((~valid).sum()), 'failure_policy':'all failed pairs count incorrect for accuracy; unavailable features never accepted',
            'threshold_protocol':'train folds only; unchanged official pair order',
            'note':'FAR estimates at very low rates have limited resolution in a 6000-pair benchmark.'}


def run(args):
    if args.backend=='sface':
        return run_sface(args)
    import torch
    torch.set_num_threads(args.threads)
    from PIL import Image
    from research.recognition import face_transform, load_embedding
    pairs = parse_pairs(args.root, args.pairs)
    if not args.checkpoint:
        raise ValueError('The resnet backend requires --checkpoint')
    model, payload = load_embedding(args.checkpoint)
    model.to(args.device)
    transform = face_transform()
    paths = sorted(set(path for pair in pairs for path in pair[:2]))
    features = {}
    with torch.inference_mode():
        for start in range(0, len(paths), args.batch_size):
            chunk = paths[start:start+args.batch_size]
            batch = []
            for path in chunk:
                with Image.open(path) as image:
                    batch.append(transform(image.convert('RGB')))
            embeddings = model(torch.stack(batch).to(args.device)).cpu().numpy()
            features.update(zip(chunk, embeddings))
    scores = [float(np.dot(features[a], features[b])) for a,b,_,_ in pairs]
    report = evaluate_scores(scores, [p[2] for p in pairs], [p[3] for p in pairs], args.far)
    report.update({'pairs':len(pairs), 'images':len(paths), 'checkpoint':str(args.checkpoint),
                   'pairs_sha256':hashlib.sha256(Path(args.pairs).read_bytes()).hexdigest(),
                   'checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                   'preprocessing':'RGB, resize 112x112, (x/255-0.5)/0.5; alignment must be supplied',
                   'training_classes':len(payload.get('classes', {}))})
    # Identity overlap is explicit; it does not silently change the official protocol.
    overlap = sorted(set(payload.get('classes',{})) & {p[0].parent.name for p in pairs} |
                     set(payload.get('classes',{})) & {p[1].parent.name for p in pairs})
    report['training_identity_name_overlap'] = overlap
    report['independence_warning'] = 'Identity names alone cannot prove absence of training/test overlap.'
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


def align_largest(vision,frame):
    """LFW subject policy: largest face, then closest center; no label inputs."""
    faces=vision.detect(frame)
    if len(faces)==0: raise ValueError('No face detected at fixed YuNet threshold 0.8')
    selected=min(faces,key=lambda face:(-float(face[2]*face[3]),
        float((face[0]+face[2]/2-frame.shape[1]/2)**2+(face[1]+face[3]/2-frame.shape[0]/2)**2)))
    return vision.recognizer.alignCrop(frame,selected),len(faces)


def run_sface(args):
    import cv2
    import time
    from app import Vision
    pairs=parse_pairs(args.root,args.pairs)
    vision=Vision(Path(args.models))
    cv2.setNumThreads(args.threads)
    paths=sorted({path for pair in pairs for path in pair[:2]})
    features={}; failures=[]; multi_face_images=0; start=time.perf_counter()
    for index,path in enumerate(paths):
        frame=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
        try:
            if frame is None: raise ValueError('OpenCV cannot decode image')
            if args.face_policy=='largest':
                aligned,detected_count=align_largest(vision,frame)
                if detected_count>1: multi_face_images+=1
                embedding=vision.recognizer.feature(aligned).reshape(-1)
            else:
                embedding=vision.feature(frame).reshape(-1)
            if not np.isfinite(embedding).all() or np.linalg.norm(embedding)<=0: raise ValueError('Invalid feature vector')
            features[path]=embedding/np.linalg.norm(embedding)
        except (ValueError,cv2.error) as error:
            failures.append({'image':str(path.relative_to(Path(args.root).resolve())),'reason':str(error)})
        if (index+1)%250==0:
            print(f'SFace: {index+1}/{len(paths)} images; {len(failures)} failures',flush=True)
    report={'backend':'OpenCV YuNet + SFace pretrained baseline','pairs':len(pairs),'images':len(paths),
        'successful_images':len(features),'failed_images':len(failures),'image_failure_rate':len(failures)/len(paths),
        'failures':failures,'elapsed_seconds':time.perf_counter()-start,'opencv':cv2.__version__,
        'pairs_sha256':hashlib.sha256(Path(args.pairs).read_bytes()).hexdigest(),
        'models_sha256':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in Path(args.models).glob('*.onnx') if 'sface' in path.name or 'yunet' in path.name},
        'preprocessing':f'BGR original LFW image -> YuNet threshold 0.8, face policy {args.face_policy} -> SFace alignCrop 112x112 -> normalized embedding',
        'face_policy':args.face_policy,'multi_face_images_resolved_geometrically':multi_face_images,
        'experiment_note':('Second preprocessing experiment after observing strict-single failures on incidental background faces. '
            'Largest bounding-box area, tie broken by distance to image center, independent of identity/pair labels. '
            'Retain original strict-single baseline. Independent final certification requires new held-out data.'
            if args.face_policy=='largest' else 'Original strict-single application policy applied to the LFW benchmark.'),
        'training_overlap':'Pretrained model training identities were not independently audited; external-data baseline.'}
    valid=[a in features and b in features for a,b,_,_ in pairs]
    report['failed_pairs']=sum(not value for value in valid)
    if failures and args.on_failure=='abort':
        report['status']='aborted: not all pairs can be evaluated; rerun with --on-failure count-incorrect for a conservative complete-protocol score'
    else:
        scores=[float(np.dot(features[a],features[b])) if good else 0. for (a,b,_,_),good in zip(pairs,valid)]
        report.update(evaluate_scores(scores,[p[2] for p in pairs],[p[3] for p in pairs],args.far,valid))
        report['status']='completed full 6000-pair protocol; failures included in denominator'
    destination=Path(args.output); destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({key:value for key,value in report.items() if key!='failures'},indent=2,ensure_ascii=False))
    if failures and args.on_failure=='abort':
        raise RuntimeError(f'{len(failures)} failed images; diagnostics saved to {destination}')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--pairs', required=True)
    parser.add_argument('--backend',choices=['resnet','sface'],default='resnet')
    parser.add_argument('--checkpoint')
    parser.add_argument('--models',default='models')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--on-failure',choices=['abort','count-incorrect'],default='abort')
    parser.add_argument('--face-policy',choices=['strict-single','largest'],default='strict-single')
    parser.add_argument('--output', default='runs/lfw.json')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--far', type=float, default=.001)
    run(parser.parse_args())

````


## research/optimize.py

**使用位置：** 研究训练与评估命令使用；详见 research/README.md 和 docs/code-map.md。对应PDF任务3.2–7.3。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/research/optimize.py)

SHA256：`8cb4e375f3325a21f82e8d0ea738524fc5a8b20c75959ee76ca132831812010c`

**代码定位：** `latency_ms` 第19行；`state_bytes` 第27行；`run` 第31行。

````python
"""CPU dynamic Linear quantization + ONNX export, checked against PyTorch.

python -m research.optimize --checkpoint runs/arcface/last.pt --images data/aligned --output runs/optimized
With --synthetic-smoke this verifies numerical plumbing only, not accuracy.
"""
import argparse
import copy
import hashlib
import io
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from research.recognition import EmbeddingNet, face_transform, load_embedding, seed_all


def latency_ms(function, warmup=3, repeats=10):
    for _ in range(warmup): function()
    timings=[]
    for _ in range(repeats):
        start=time.perf_counter(); function(); timings.append((time.perf_counter()-start)*1000)
    return {'median_ms':float(np.median(timings)), 'p95_ms':float(np.percentile(timings,95)), 'repeats':repeats}


def state_bytes(model):
    stream=io.BytesIO(); torch.save(model.state_dict(),stream); return stream.tell()


def run(args):
    import onnx
    import onnxruntime as ort
    if min(args.threads,args.samples,args.repeats)<1: raise ValueError('threads, samples and repeats must be positive')
    if bool(args.lfw_root)!=bool(args.pairs): raise ValueError('LFW requires both --lfw-root and --pairs')
    if args.synthetic_smoke and args.pairs: raise ValueError('Synthetic smoke must not be used as a benchmark model')
    seed_all(42); torch.set_num_threads(args.threads)
    checkpoint_payload=None
    if args.synthetic_smoke:
        if args.checkpoint: raise ValueError('Do not mix synthetic smoke and a trained checkpoint')
        model=EmbeddingNet(128).eval(); batch=torch.randn(2,3,112,112)
        provenance='SYNTHETIC CODE SMOKE ONLY: random weights and inputs; no recognition result'
    else:
        if not args.checkpoint or not args.images: raise ValueError('Supply checkpoint + aligned images, or explicitly --synthetic-smoke')
        model,checkpoint_payload=load_embedding(args.checkpoint)
        paths=sorted(p for p in Path(args.images).rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png'})[:args.samples]
        if not paths: raise ValueError('No representative aligned images')
        images=[]
        for path in paths:
            with Image.open(path) as image: images.append(face_transform()(image.convert('RGB')))
        batch=torch.stack(images)
        provenance=f'Checkpoint {args.checkpoint}; {len(paths)} representative aligned images; no accuracy benchmark unless pairs supplied'
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    # ponytail: dynamic quantization covers Linear only; Conv2d dominates ResNet.
    # This cannot be advertised as a 4x smaller/faster full int8 convolution model.
    quantized=torch.ao.quantization.quantize_dynamic(copy.deepcopy(model),{torch.nn.Linear},dtype=torch.qint8)
    with torch.inference_mode():
        reference=model(batch).numpy(); int8_result=quantized(batch).numpy()
        float_latency=latency_ms(lambda:model(batch),repeats=args.repeats)
        int8_latency=latency_ms(lambda:quantized(batch),repeats=args.repeats)
        path=output/'embedding.onnx'
        torch.onnx.export(model,batch[:1],str(path),input_names=['images'],output_names=['embedding'],
            dynamic_axes={'images':{0:'batch'},'embedding':{0:'batch'}},opset_version=17,dynamo=False)
    onnx.checker.check_model(str(path))
    options=ort.SessionOptions(); options.intra_op_num_threads=args.threads
    session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    onnx_result=session.run(None,{'images':batch.numpy()})[0]
    np.testing.assert_allclose(reference,onnx_result,rtol=1e-3,atol=1e-5)
    report={'provenance':provenance,'torch':torch.__version__,'onnx_version':onnx.__version__,'onnxruntime':ort.__version__,
        'cpu_threads':args.threads,'batch_size':len(batch),'fp32':{'state_bytes':state_bytes(model),**float_latency},
        'dynamic_linear_int8':{'state_bytes':state_bytes(quantized),**int8_latency,
            'embedding_max_abs_error':float(np.max(np.abs(reference-int8_result))),
            'mean_cosine_to_fp32':float(np.sum(reference*int8_result,axis=1).mean())},
        'onnx':{'bytes':path.stat().st_size,'embedding_max_abs_error':float(np.max(np.abs(reference-onnx_result))),
            **latency_ms(lambda:session.run(None,{'images':batch.numpy()}),repeats=args.repeats)},
        'accuracy':None, 'accuracy_note':'Numerical embedding similarity is not face verification accuracy.'}
    if args.checkpoint:
        with Path(args.checkpoint).open('rb') as file: report['checkpoint_sha256']=hashlib.file_digest(file,'sha256').hexdigest()
    if args.pairs:
        from research.lfw import parse_pairs,evaluate_scores
        pairs=parse_pairs(args.lfw_root,args.pairs)
        features=[{},{}]
        paths=sorted({p for pair in pairs for p in pair[:2]})
        with torch.inference_mode():
            for start in range(0,len(paths),32):
                chunk=paths[start:start+32]; tensors=[]
                for image_path in chunk:
                    with Image.open(image_path) as image: tensors.append(face_transform()(image.convert('RGB')))
                batch_images=torch.stack(tensors)
                for index,current in enumerate([model,quantized]): features[index].update(zip(chunk,current(batch_images).numpy()))
                if start%512==0 or start+32>=len(paths): print(f'FP32/int8 LFW embeddings {min(start+32,len(paths))}/{len(paths)}',flush=True)
        report['accuracy']={name:evaluate_scores([np.dot(embedding[a],embedding[b]) for a,b,_,_ in pairs],
            [p[2] for p in pairs],[p[3] for p in pairs]) for name,embedding in zip(['fp32','dynamic_linear_int8'],features)}
        report['accuracy_note']='Official pairs and training-fold threshold calibration for each model independently.'
        report['lfw_pairs_sha256']=hashlib.sha256(Path(args.pairs).read_bytes()).hexdigest()
        report['lfw_pairs']=len(pairs); report['lfw_unique_images']=len(paths)
        report['training_identity_name_overlap']=sorted(set(checkpoint_payload.get('classes',{})) & {p.parent.name for p in paths})
    torch.save({'model_type':'resnet50_arcface_dynamic_linear_int8','model':quantized.state_dict(),
        'embedding_dim':model.backbone.fc.out_features,'provenance':provenance},output/'embedding_dynamic_int8.pt')
    (output/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2)); return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint'); parser.add_argument('--images'); parser.add_argument('--output',default='runs/optimized')
    parser.add_argument('--synthetic-smoke',action='store_true'); parser.add_argument('--samples',type=int,default=8)
    parser.add_argument('--repeats',type=int,default=10); parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--lfw-root'); parser.add_argument('--pairs')
    run(parser.parse_args())

````


## research/prepare_lfw_pilot.py

**使用位置：** 研究训练与评估命令使用；详见 research/README.md 和 docs/code-map.md。对应PDF任务3.2–7.3。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/research/prepare_lfw_pilot.py)

SHA256：`3a305e6b6c37db9f555dcde2bd5f738d7068c4f742a3ca73274a2477bdddb508`

**代码定位：** `run` 第17行。

````python
"""Prepare real, identity-disjoint LFW training pilot and consistently aligned test.

Training uses only identities never mentioned in the official 6000 pairs, with at
least two original photographs each. This is NOT MS-Celeb-1M or its substitute
for formal task acceptance; it is a small, reproducible available-data experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from app import Vision
from research.lfw import align_largest,parse_pairs


def run(args):
    root=Path(args.root).resolve(); pairs=parse_pairs(root,args.pairs)
    evaluation_paths=sorted({path for pair in pairs for path in pair[:2]})
    evaluation_names={path.parent.name for path in evaluation_paths}
    folders=sorted(path for path in root.iterdir() if path.is_dir() and path.name not in evaluation_names and len(list(path.glob('*.jpg')))>=2)
    training_paths=sorted(path for folder in folders for path in folder.glob('*.jpg'))
    if not training_paths: raise ValueError('No disjoint identities with >=2 images are available')
    train_output=Path(args.train_output).resolve(); eval_output=Path(args.eval_output).resolve()
    if any(a.is_relative_to(b) or b.is_relative_to(a) for a,b in [(train_output,eval_output),(train_output,root),(eval_output,root)]):
        raise ValueError('Training, evaluation and original images need separate non-overlapping locations')
    vision=Vision(Path(args.models)); cv2.setNumThreads(args.threads)
    report={'scope':'Real LFW identity-disjoint small training pilot; not MS-Celeb-1M training',
        'training_identities':[path.name for path in folders],'training_images':len(training_paths),
        'evaluation_identities':len(evaluation_names),'evaluation_images':len(evaluation_paths),
        'identity_overlap':sorted({path.name for path in folders}&evaluation_names),
        'pairs_sha256':hashlib.sha256(Path(args.pairs).read_bytes()).hexdigest(),
        'alignment':'YuNet .8; largest face, center-distance tie; SFace 5-point similarity alignCrop; 112x112 JPEG quality100 for both sets',
        'training_sources':[],'failures':[]}
    for split,paths,destination in [('train',training_paths,train_output),('evaluation',evaluation_paths,eval_output)]:
        aggregate=hashlib.sha256()
        for index,path in enumerate(paths):
            source=path.read_bytes(); relative=path.relative_to(root); source_hash=hashlib.sha256(source).hexdigest()
            aggregate.update(str(relative).replace('\\','/').encode('utf-8')); aggregate.update(source_hash.encode('ascii'))
            frame=cv2.imdecode(np.frombuffer(source,np.uint8),cv2.IMREAD_COLOR)
            try:
                if frame is None: raise ValueError('Cannot decode image')
                aligned,count=align_largest(vision,frame)
                success,encoded=cv2.imencode('.jpg',aligned,[cv2.IMWRITE_JPEG_QUALITY,100])
                if not success: raise ValueError('Cannot encode aligned crop')
                output=destination/relative; output.parent.mkdir(parents=True,exist_ok=True); encoded.tofile(output)
                if split=='train': report['training_sources'].append({'path':str(relative),'original_sha256':source_hash,
                    'aligned_sha256':hashlib.sha256(encoded.tobytes()).hexdigest(),'detected_faces':count})
            except (ValueError,cv2.error) as error:
                report['failures'].append({'split':split,'image':str(relative),'reason':str(error)})
            if (index+1)%500==0 or index+1==len(paths): print(f'Align {split}: {index+1}/{len(paths)}; failures={len(report["failures"])}',flush=True)
        report[split+'_ordered_source_digest']=aggregate.hexdigest()
    destination=Path(args.report); destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2),encoding='utf-8')
    if report['failures']: raise RuntimeError('Alignment failures recorded; refusing to claim complete prepared splits')
    print(json.dumps({key:value for key,value in report.items() if key not in {'training_sources','training_identities'}},indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='data/lfw/lfw'); parser.add_argument('--pairs',default='data/lfw/pairs.txt')
    parser.add_argument('--models',default='models'); parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--train-output',default='data/lfw-pilot-train'); parser.add_argument('--eval-output',default='data/lfw-aligned')
    parser.add_argument('--report',default='reports/lfw-pilot-data.json')
    run(parser.parse_args())

````


## research/recognition.py

**使用位置：** 研究训练与评估命令使用；详见 research/README.md 和 docs/code-map.md。对应PDF任务3.2–7.3。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/research/recognition.py)

SHA256：`be2044741b7a0ed7ce59eac473a88ab586ad2d2c529e6d924c237007c06c2893`

**代码定位：** `seed_all` 第23行；`face_transform` 第33行；`EmbeddingNet` 第40行；`ArcFace` 第50行；`PKSampler` 第70行；`batch_hard_triplet` 第93行；`load_embedding` 第107行；`save_curves` 第116行；`train` 第143行。

````python
"""ResNet50 + ArcFace; train on locally licensed, aligned identity folders.

python -m research.recognition --data data/identities --output runs/arcface
Each identity directory must contain >=2 images. Inputs are RGB faces, 112x112,
normalized to [-1,1]. Random initialization is intentional; LFW is evaluation only.
"""
import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets, models, transforms


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def face_transform(train=False, size=112):
    steps = [transforms.Resize((size, size))]
    if train:
        steps.append(transforms.RandomHorizontalFlip())
    return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize([.5]*3, [.5]*3)])


class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embedding_dim)

    def forward(self, images):
        return F.normalize(self.backbone(images), dim=1)


class ArcFace(nn.Module):
    """Additive angular margin: s*cos(theta_y+m), monotonic fallback near pi."""
    def __init__(self, embedding_dim, classes, margin=.5, scale=64.):
        super().__init__()
        if classes < 2 or not 0 < margin < math.pi/2 or scale <= 0:
            raise ValueError('ArcFace needs >=2 classes, margin in (0,pi/2), scale >0')
        self.weight = nn.Parameter(torch.empty(classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin, self.scale = margin, scale

    def forward(self, features, labels):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight)).clamp(-1+1e-7, 1-1e-7)
        sine = torch.sqrt((1-cosine.square()).clamp_min(1e-7))
        phi = cosine*math.cos(self.margin)-sine*math.sin(self.margin)
        phi = torch.where(cosine > math.cos(math.pi-self.margin), phi,
                          cosine-math.sin(math.pi-self.margin)*self.margin)
        hot = F.one_hot(labels, self.weight.shape[0]).to(cosine.dtype)
        return self.scale * (hot*phi + (1-hot)*cosine)


class PKSampler(Sampler):
    """P different identities, K different images per identity; epoch-seeded."""
    def __init__(self, labels, p=8, k=4, seed=42):
        self.groups = defaultdict(list)
        for index, label in enumerate(labels):
            self.groups[label].append(index)
        if p < 2 or k < 2 or len(self.groups) < p:
            raise ValueError('Need P>=2, K>=2 and at least P identities')
        if any(len(indices) < k for indices in self.groups.values()):
            raise ValueError('Every identity needs at least K distinct images; reduce K or clean data')
        self.p, self.k, self.seed, self.epoch = p, k, seed, 0
        self.batches = max(1, math.ceil(len(labels)/(p*k)))

    def __iter__(self):
        rng = random.Random(self.seed+self.epoch)
        for _ in range(self.batches):
            yield [index for label in rng.sample(list(self.groups), self.p)
                   for index in rng.sample(self.groups[label], self.k)]

    def __len__(self):
        return self.batches


def batch_hard_triplet(features, labels, margin=.2):
    """Hardest positive and negative for each valid anchor in its P-K batch."""
    distances = torch.cdist(features, features)
    same = labels[:, None].eq(labels[None, :])
    positive = same & ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    negative = ~same
    valid = positive.any(1) & negative.any(1)
    if not valid.any():
        raise ValueError('Batch-hard mining needs positive and negative examples per anchor')
    hardest_positive = distances.masked_fill(~positive, float('-inf')).amax(1)
    hardest_negative = distances.masked_fill(~negative, float('inf')).amin(1)
    return F.relu(hardest_positive[valid]-hardest_negative[valid]+margin).mean()


def load_embedding(checkpoint):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if payload.get('model_type') != 'resnet50_arcface':
        raise ValueError('Expected a research.recognition checkpoint')
    model = EmbeddingNet(payload['embedding_dim'])
    model.load_state_dict(payload['model'])
    return model.eval(), payload


def save_curves(history, output):
    """Dependency-free SVG of measured epoch loss and training classification rate."""
    with Path(history).open(encoding='utf-8',newline='') as file: rows=list(csv.DictReader(file))
    if not rows: raise ValueError('No history rows to plot')
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="400" viewBox="0 0 1100 400">',
        '<rect width="1100" height="400" fill="#f8fafc"/>',
        '<text x="40" y="30" font-family="sans-serif" font-size="20">Measured training history (not verification accuracy)</text>']
    for panel,(column,title) in enumerate([('loss','ArcFace + batch-hard loss'),('margin_classifier_train_accuracy','Margin classifier training accuracy')]):
        values=[float(row[column]) for row in rows]; left=55+panel*540; top=75; width=455; height=255
        low=0.; high=max(values)*1.1 if column=='loss' else 1.
        high=max(high,1e-6)
        points=[(left+index*width/max(1,len(rows)-1),top+height-height*(value-low)/(high-low)) for index,value in enumerate(values)]
        parts.extend([f'<text x="{left}" y="55" font-family="sans-serif" font-size="16">{title}</text>',
            f'<path d="M{left},{top} V{top+height} H{left+width}" fill="none" stroke="#64748b"/>'])
        for tick in range(5):
            y=top+height-tick*height/4
            parts.extend([f'<path d="M{left},{y} H{left+width}" stroke="#dbe3eb"/>',
                f'<text x="{left-8}" y="{y+4}" text-anchor="end" font-family="sans-serif" font-size="11">{high*tick/4:.3g}</text>'])
        parts.append('<polyline points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in points)+'" fill="none" stroke="#2563eb" stroke-width="3"/>')
        for row,(x,y),value in zip(rows,points,values):
            parts.extend([f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#2563eb"/>',
                f'<text x="{x:.2f}" y="{y-9:.2f}" text-anchor="middle" font-family="sans-serif" font-size="12">{value:.4f}</text>',
                f'<text x="{x:.2f}" y="{top+height+20}" text-anchor="middle" font-family="sans-serif" font-size="12">{int(row["epoch"])}</text>'])
        parts.append(f'<text x="{left+width/2}" y="385" text-anchor="middle" font-family="sans-serif" font-size="13">Epoch</text>')
    Path(output).write_text('\n'.join(parts+['</svg>']),encoding='utf-8')


def train(args):
    seed_all(args.seed)
    torch.set_num_threads(getattr(args,'threads',2))
    if args.epochs < 1 or args.lr <= 0 or args.triplet_weight < 0:
        raise ValueError('epochs and lr must be positive; triplet weight nonnegative')
    dataset = datasets.ImageFolder(args.data, transform=face_transform(True))
    sampler = PKSampler(dataset.targets, args.p, args.k, args.seed)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers)
    device = torch.device(args.device)
    model, head = EmbeddingNet(args.embedding_dim).to(device), ArcFace(args.embedding_dim, len(dataset.classes)).to(device)
    optimizer = torch.optim.SGD(list(model.parameters())+list(head.parameters()), lr=args.lr, momentum=.9, weight_decay=5e-4)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'classes.json').write_text(json.dumps(dataset.class_to_idx, indent=2), encoding='utf-8')
    (output/'config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    with (output/'history.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'loss', 'margin_classifier_train_accuracy', 'samples'])
        for epoch in range(args.epochs):
            sampler.epoch = epoch
            model.train(); head.train()
            total_loss, correct, count = 0., 0, 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                features = model(images)
                logits = head(features, labels)
                loss = F.cross_entropy(logits, labels) + args.triplet_weight*batch_hard_triplet(features, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite training loss')
                loss.backward()
                nn.utils.clip_grad_norm_(list(model.parameters())+list(head.parameters()), 5.)
                optimizer.step()
                total_loss += float(loss.detach())*len(labels)
                correct += int((logits.argmax(1)==labels).sum())
                count += len(labels)
            row = [epoch+1, total_loss/count, correct/count, count]
            writer.writerow(row); file.flush()
            print(dict(zip(['epoch','loss','train_accuracy','samples'], row)), flush=True)
            torch.save({'model_type':'resnet50_arcface', 'embedding_dim':args.embedding_dim,
                        'model':model.state_dict(), 'head':head.state_dict(),
                        'optimizer':optimizer.state_dict(), 'epoch':epoch+1,
                        'classes':dataset.class_to_idx, 'seed':args.seed}, output/'last.pt')
    save_curves(output/'history.csv',output/'training-curves.svg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default='runs/arcface')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--p', type=int, default=8)
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--embedding-dim', type=int, default=512)
    parser.add_argument('--lr', type=float, default=.05)
    parser.add_argument('--triplet-weight', type=float, default=.1)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    train(parser.parse_args())

````


## research/smoke.py

**使用位置：** 研究训练与评估命令使用；详见 research/README.md 和 docs/code-map.md。对应PDF任务3.2–7.3。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/research/smoke.py)

SHA256：`aac1026379764557956ae0262144b7860fbe7eeea03213b37d63f885b537b488`

**代码定位：** `end_to_end` 第13行；`run` 第53行。

````python
"""Random-tensor gradient checks only. These are NOT training or benchmark results."""
import argparse
import json
import tempfile
from types import SimpleNamespace
from pathlib import Path
import torch
from research.recognition import ArcFace,EmbeddingNet,PKSampler,batch_hard_triplet,seed_all
from research.landmarks import make_model
from research.stargan import Generator,Discriminator,gradient_penalty


def end_to_end():
    """Exercise loaders/optimizers/checkpoints on explicit throwaway noise fixtures."""
    import numpy as np
    from PIL import Image
    from research.recognition import train as train_recognition,load_embedding
    from research.landmarks import train as train_landmarks,infer as infer_landmarks
    from research.stargan import train as train_gan,generate
    with tempfile.TemporaryDirectory(prefix='face-vision-synthetic-smoke-') as directory:
        root=Path(directory); rng=np.random.default_rng(42)
        for identity in ['synthetic_A','synthetic_B']:
            (root/'identities'/identity).mkdir(parents=True)
            for number in range(2):
                Image.fromarray(rng.integers(0,256,(178,178,3),dtype=np.uint8)).save(root/'identities'/identity/f'{number}.jpg')
        train_recognition(SimpleNamespace(data=str(root/'identities'),output=str(root/'arcface'),
            epochs=1,p=2,k=2,seed=42,workers=0,embedding_dim=32,device='cpu',lr=.001,triplet_weight=.1))
        model,_=load_embedding(root/'arcface/last.pt')
        with torch.inference_mode(): assert model(torch.randn(1,3,112,112)).shape==(1,32)
        entries=[]
        for number,path in enumerate(sorted((root/'identities').rglob('*.jpg'))):
            points=np.column_stack([np.linspace(30,140,68),80+20*np.sin(np.arange(68))])
            entries.append({'image':str(path.relative_to(root)),'points':points.tolist(),'box':[0,0,178,178]})
        for name,subset in [('train',entries[:2]),('val',entries[2:])]:
            (root/f'{name}.json').write_text(json.dumps(subset),encoding='utf-8')
        train_landmarks(SimpleNamespace(root=str(root),val_root=None,train=str(root/'train.json'),val=str(root/'val.json'),
            output=str(root/'landmark-run'),epochs=1,batch_size=2,seed=42,device='cpu',lr=.001))
        first=root/entries[0]['image']
        infer_landmarks(SimpleNamespace(checkpoint=root/'landmark-run/last.pt',image=first,box=None,output=root/'landmark-result'))
        assert (root/'landmark-result/aligned.png').is_file()
        (root/'celeba').mkdir()
        for index in range(2):
            Image.fromarray(rng.integers(0,256,(178,178,3),dtype=np.uint8)).save(root/'celeba'/f'{index:06d}.jpg')
        (root/'attributes.txt').write_text('2\nBlack_Hair Young\n000000.jpg 1 -1\n000001.jpg -1 1\n')
        (root/'partition.txt').write_text('000000.jpg 0\n000001.jpg 0\n')
        train_gan(SimpleNamespace(root=str(root/'celeba'),labels=str(root/'attributes.txt'),partition=str(root/'partition.txt'),
            attributes=['Black_Hair','Young'],size=64,steps=1,n_critic=1,batch_size=2,width=8,blocks=1,lr=.0001,
            seed=42,device='cpu',output=str(root/'gan'),log_interval=1,save_interval=1))
        generate(SimpleNamespace(checkpoint=root/'gan/last.pt',image=root/'celeba/000000.jpg',targets='0,1',output=root/'translation.png'))
        assert (root/'translation.png').is_file()


def run(output, full=False):
    seed_all(42); torch.set_num_threads(2)
    model=EmbeddingNet(32).train(); head=ArcFace(32,2)
    images=torch.randn(4,3,64,64); labels=torch.tensor([0,0,1,1])
    optimizer=torch.optim.SGD(list(model.parameters())+list(head.parameters()),lr=.001)
    before=head.weight.detach().clone(); features=model(images)
    loss=torch.nn.functional.cross_entropy(head(features,labels),labels)+batch_hard_triplet(features,labels)
    loss.backward(); optimizer.step()
    assert torch.isfinite(loss) and not torch.equal(before,head.weight.detach())
    assert list(PKSampler([0,0,1,1],2,2))
    landmark_model=make_model().train()
    landmarks=landmark_model(images).reshape(4,68,2)
    landmarks.square().mean().backward()
    assert torch.isfinite(landmarks).all()
    generator=Generator(3,width=8,blocks=1); discriminator=Discriminator(3,width=8,depth=3)
    real=torch.randn(2,3,32,32); attributes=torch.tensor([[1.,0.,1.],[0.,1.,0.]])
    fake=generator(real,attributes)
    assert fake.shape==real.shape
    score,classification=discriminator(fake)
    gan_loss=-score.mean()+torch.nn.functional.binary_cross_entropy_with_logits(classification,attributes)
    gan_loss.backward()
    penalty=gradient_penalty(discriminator,real,fake.detach()); penalty.backward()
    assert torch.isfinite(penalty) and any(p.grad is not None for p in generator.parameters())
    if full: end_to_end()
    report={'status':'passed','scope':'SYNTHETIC CODE SMOKE ONLY; random tensors and untrained models',
        'torch':torch.__version__,'checks':['ResNet50/ArcFace forward-backward and optimizer update',
        'P-K sampler','batch-hard triplet gradients','68 landmark regressor gradients','conditional generator shape',
        'discriminator/classifier and WGAN gradient-penalty backward'],
        'benchmark_accuracy':None,'trained_face_model':False,
        'end_to_end_loaders_checkpoints_inference':full,
        'fixture_note':'All temporary synthetic fixtures/checkpoints are deleted after the smoke check.'}
    path=Path(output); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',default='runs/research-smoke.json')
    parser.add_argument('--end-to-end',action='store_true')
    args=parser.parse_args(); run(args.output,args.end_to_end)

````


## research/stargan.py

**使用位置：** 研究训练与评估命令使用；详见 research/README.md 和 docs/code-map.md。对应PDF任务3.2–7.3。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/research/stargan.py)

SHA256：`bc425a69d23af169f10c1b621068ef327a85b0abea80caf7d4d8f26cc5ec98d7`

**代码定位：** `Residual` 第20行；`Generator` 第29行；`Discriminator` 第45行；`CelebA` 第60行；`gradient_penalty` 第96行；`train` 第104行；`generate` 第152行；`metrics` 第169行。

````python
"""Small, actual StarGAN-style WGAN-GP training/translation/FID+IS entry points.

This educational implementation is not pretrained and does not imply a successful
attribute editor until real CelebA training and held-out inspection are complete.
"""
import argparse
import csv
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from research.recognition import seed_all


class Residual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(width,width,3,1,1,bias=False), nn.InstanceNorm2d(width,affine=True),
            nn.ReLU(), nn.Conv2d(width,width,3,1,1,bias=False), nn.InstanceNorm2d(width,affine=True))

    def forward(self,x): return x+self.block(x)


class Generator(nn.Module):
    def __init__(self, attributes, width=64, blocks=6):
        super().__init__()
        layers=[nn.Conv2d(3+attributes,width,7,1,3,bias=False),nn.InstanceNorm2d(width,affine=True),nn.ReLU()]
        for _ in range(2):
            layers.extend([nn.Conv2d(width,width*2,4,2,1,bias=False),nn.InstanceNorm2d(width*2,affine=True),nn.ReLU()]); width*=2
        layers.extend([Residual(width) for _ in range(blocks)])
        for _ in range(2):
            layers.extend([nn.ConvTranspose2d(width,width//2,4,2,1,bias=False),nn.InstanceNorm2d(width//2,affine=True),nn.ReLU()]); width//=2
        self.network=nn.Sequential(*layers,nn.Conv2d(width,3,7,1,3),nn.Tanh())

    def forward(self,image,attributes):
        conditions=attributes[:,:,None,None].expand(-1,-1,image.shape[2],image.shape[3])
        return self.network(torch.cat([image,conditions],dim=1))


class Discriminator(nn.Module):
    def __init__(self,attributes,width=64,depth=5):
        super().__init__()
        layers=[]; channels=3
        for _ in range(depth):
            layers.extend([nn.Conv2d(channels,width,4,2,1),nn.LeakyReLU(.01)]); channels=width; width*=2
        self.features=nn.Sequential(*layers)
        self.source=nn.Conv2d(channels,1,3,1,1)
        self.classifier=nn.Linear(channels,attributes)

    def forward(self,image):
        features=self.features(image)
        return self.source(features).mean((1,2,3)),self.classifier(features.mean((2,3)))


class CelebA(Dataset):
    def __init__(self,root,attributes_file,partition_file,attributes,size):
        self.root=Path(root).resolve()
        lines=Path(attributes_file).read_text(encoding='utf-8').strip().splitlines()
        total=int(lines[0]); names=lines[1].split()
        if len(lines)-2!=total: raise ValueError('CelebA attribute count does not match header')
        if len(set(attributes))!=len(attributes) or not set(attributes)<=set(names): raise ValueError('Unknown/duplicate CelebA attributes')
        columns=[names.index(name) for name in attributes]
        partitions={}
        for line in Path(partition_file).read_text(encoding='utf-8').splitlines():
            name,split=line.split()
            if name in partitions or split not in {'0','1','2'}: raise ValueError('Invalid/duplicate partition entry')
            partitions[name]=split
        self.entries=[]; seen=set()
        for line in lines[2:]:
            fields=line.split(); name=fields[0]
            if name in seen or len(fields)!=len(names)+1 or any(value not in {'-1','1'} for value in fields[1:]):
                raise ValueError('Malformed CelebA attribute row')
            seen.add(name)
            if name not in partitions: raise ValueError('Every CelebA image needs a partition entry')
            if partitions[name]!='0': continue
            path=(self.root/name).resolve()
            if not path.is_relative_to(self.root) or not path.is_file(): raise FileNotFoundError(f'Missing training image: {path}')
            self.entries.append((path,torch.tensor([float(fields[column+1]=='1') for column in columns])))
        if len(self.entries)<2: raise ValueError('Need >=2 real CelebA training images')
        self.transform=transforms.Compose([transforms.CenterCrop(178),transforms.Resize((size,size)),
            transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])

    def __len__(self): return len(self.entries)

    def __getitem__(self,index):
        path,labels=self.entries[index]
        with Image.open(path) as image: tensor=self.transform(image.convert('RGB'))
        return tensor,labels


def gradient_penalty(discriminator,real,fake):
    alpha=torch.rand(real.shape[0],1,1,1,device=real.device)
    mixed=(alpha*real+(1-alpha)*fake).requires_grad_(True)
    scores,_=discriminator(mixed)
    gradient=torch.autograd.grad(scores.sum(),mixed,create_graph=True)[0]
    return (gradient.flatten(1).norm(2,dim=1)-1).square().mean()


def train(args):
    seed_all(args.seed)
    if args.size not in {64,128,256} or args.n_critic<1 or args.steps<args.n_critic or args.batch_size<2 or min(args.width,args.blocks,args.lr,args.log_interval,args.save_interval)<=0:
        raise ValueError('Use size 64/128/256, steps>=n_critic>=1, batch_size>=2, positive model/optimizer/logging settings')
    dataset=CelebA(args.root,args.labels,args.partition,args.attributes,args.size)
    loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True,drop_last=True)
    if not len(loader): raise ValueError('Batch size exceeds training data size')
    generator=Generator(len(args.attributes),args.width,args.blocks).to(args.device)
    discriminator=Discriminator(len(args.attributes),args.width).to(args.device)
    g_optimizer=torch.optim.Adam(generator.parameters(),lr=args.lr,betas=(.5,.999))
    d_optimizer=torch.optim.Adam(discriminator.parameters(),lr=args.lr,betas=(.5,.999))
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    iterator=iter(loader)
    with (output/'history.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.writer(file); writer.writerow(['step','d_loss','g_loss','reconstruction_l1'])
        for step in range(1,args.steps+1):
            try: real,original=next(iterator)
            except StopIteration: iterator=iter(loader); real,original=next(iterator)
            real,original=real.to(args.device),original.to(args.device)
            target=original[torch.randperm(len(original),device=args.device)]
            d_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(): fake=generator(real,target)
            real_score,real_class=discriminator(real); fake_score,_=discriminator(fake)
            d_loss=fake_score.mean()-real_score.mean()+F.binary_cross_entropy_with_logits(real_class,original)+10*gradient_penalty(discriminator,real,fake)
            if not torch.isfinite(d_loss): raise FloatingPointError('Non-finite discriminator loss')
            d_loss.backward(); d_optimizer.step()
            g_loss=None; reconstruction=None
            if step%args.n_critic==0:
                for parameter in discriminator.parameters(): parameter.requires_grad_(False)
                g_optimizer.zero_grad(set_to_none=True)
                fake=generator(real,target); score,classification=discriminator(fake)
                reconstruction=F.l1_loss(generator(fake,original),real)
                g_loss=-score.mean()+F.binary_cross_entropy_with_logits(classification,target)+10*reconstruction
                if not torch.isfinite(g_loss): raise FloatingPointError('Non-finite generator loss')
                g_loss.backward(); g_optimizer.step()
                for parameter in discriminator.parameters(): parameter.requires_grad_(True)
            if step%args.log_interval==0 or step==args.steps:
                row=[step,d_loss.item(),None if g_loss is None else g_loss.item(),None if reconstruction is None else reconstruction.item()]
                writer.writerow(row); file.flush(); print(row,flush=True)
            if step%args.save_interval==0 or step==args.steps:
                torch.save({'model_type':'educational_stargan','generator':generator.state_dict(),
                    'discriminator':discriminator.state_dict(),'g_optimizer':g_optimizer.state_dict(),
                    'd_optimizer':d_optimizer.state_dict(),'attributes':args.attributes,'width':args.width,
                    'blocks':args.blocks,'size':args.size,'step':step,'seed':args.seed},output/'last.pt')
                with torch.no_grad():
                    save_image(torch.cat([real[:4],generator(real[:4],target[:4])]),output/'latest_grid.png',nrow=min(4,len(real)),normalize=True,value_range=(-1,1))


def generate(args):
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if checkpoint.get('model_type')!='educational_stargan': raise ValueError('Expected this implementation checkpoint')
    targets=[int(value) for value in args.targets.split(',')]
    if len(targets)!=len(checkpoint['attributes']) or not set(targets)<={0,1}: raise ValueError('targets must have one 0/1 per checkpoint attribute')
    hair=[targets[index] for index,name in enumerate(checkpoint['attributes']) if name in {'Black_Hair','Blond_Hair','Brown_Hair','Gray_Hair'}]
    if sum(hair)>1: raise ValueError('Choose at most one mutually exclusive hair color')
    generator=Generator(len(targets),checkpoint['width'],checkpoint['blocks']).eval()
    generator.load_state_dict(checkpoint['generator'])
    with Image.open(args.image) as image:
        transform=transforms.Compose([transforms.CenterCrop(178),transforms.Resize((checkpoint['size'],)*2),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])
        tensor=transform(image.convert('RGB'))[None]
    destination=Path(args.output); destination.parent.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode(): result=generator(tensor,torch.tensor([targets],dtype=torch.float32))
    save_image(result,destination,normalize=True,value_range=(-1,1))


def metrics(args):
    import torch_fidelity
    counts=[]
    for directory in [args.real,args.generated]:
        files=[p for p in Path(directory).rglob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg'}]
        if len(files)<10: raise ValueError('Need at least ten images per set for ten IS splits; use thousands of held-out samples for meaningful estimates')
        counts.append(len(files))
    result=torch_fidelity.calculate_metrics(input1=args.generated,input2=args.real,cuda=args.cuda,
        isc=True,fid=True,kid=False,verbose=True)
    result['note']='Report image count/preprocessing/sampling policy; small-sample FID and IS are unreliable. IS is ImageNet-based and limited for faces.'
    result.update({'real_images':counts[0],'generated_images':counts[1],'real_directory':str(args.real),'generated_directory':str(args.generated)})
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('train')
    for name in ['root','labels','partition']: p.add_argument('--'+name,required=True)
    p.add_argument('--attributes',nargs='+',default=['Black_Hair','Blond_Hair','Brown_Hair','Male','Young'])
    p.add_argument('--output',default='runs/stargan'); p.add_argument('--device',default='cpu')
    for name,value in [('steps',100000),('batch-size',8),('size',128),('width',64),('blocks',6),('n-critic',5),('seed',42),('log-interval',10),('save-interval',1000)]: p.add_argument('--'+name,type=int,default=value)
    p.add_argument('--lr',type=float,default=.0001)
    p=commands.add_parser('generate')
    for name in ['checkpoint','image','targets','output']: p.add_argument('--'+name,required=True)
    p=commands.add_parser('metrics')
    for name in ['real','generated']: p.add_argument('--'+name,required=True)
    p.add_argument('--output',default='runs/stargan-metrics.json'); p.add_argument('--cuda',action='store_true')
    args=parser.parse_args(); {'train':train,'generate':generate,'metrics':metrics}[args.command](args)

````


## scripts/build_deliverables.py

**使用位置：** 模型下载、环境检查、数据准备或交付物生成；本文件开头docstring和下方函数索引提供具体入口。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/scripts/build_deliverables.py)

SHA256：`f8507961c2bcf108aa3f29092c9495a88502b6a70a5fa6a51e81d80e24a260a8`

**代码定位：** `rendered` 第35行；`make_html` 第40行；`source_files` 第46行；`use_of` 第54行；`make_compendium` 第69行；`pdf_from_markdown` 第104行；`slides` 第151行；`main` 第201行。

````python
"""Generate professional tutorial PDFs, slides, and a complete code/use-site compendium.

Run with the project research/document environment. Reads source, never edits it.
"""
from __future__ import annotations
import ast
import hashlib
import html
import json
import re
import textwrap
from pathlib import Path

import markdown
from bs4 import BeautifulSoup
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'
REPORTS = ROOT / 'reports'
OUTPUT = ROOT.parent

CSS = '''body{font:17px/1.85 "Segoe UI","Microsoft YaHei",sans-serif;color:#1b2925;background:#f7f8f4;max-width:1040px;margin:auto;padding:44px 28px}h1{font-size:36px;line-height:1.4;color:#1c4939}h2{font-size:25px;margin-top:2.5em;border-bottom:1px solid #cdd7cf;padding-bottom:12px}h3{font-size:20px;margin-top:1.8em}a{color:#1f6550}p{margin:1em 0}code{font:14px/1.65 Consolas,"Microsoft YaHei",monospace;background:#eaf0ea;padding:2px 4px}pre{background:#142c25;color:#e4efdf;padding:22px;overflow:auto;border-radius:8px;line-height:1.7}pre code{background:transparent;color:inherit;white-space:pre}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:12px;border:1px solid #d4ded6;vertical-align:top;overflow-wrap:anywhere}th{background:#e7eee8;text-align:left}.toc{font-size:14px;columns:2;column-gap:36px}.toc ul{padding-left:18px}blockquote{margin-left:0;padding-left:18px;border-left:3px solid #789c85;color:#40594b}img{max-width:100%}.meta{font-size:13px;color:#63756a}@media(max-width:600px){.toc{columns:1}body{padding:20px 16px}h1{font-size:28px}table{display:block;overflow:auto}}@media print{body{background:white;max-width:none;font-size:10pt;padding:0}.toc{columns:2}h2{break-before:page}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f2;color:#111;font-size:8pt}pre code{white-space:pre-wrap}a{color:inherit}table{font-size:8pt}tr{break-inside:avoid}}'''


def rendered(source: str) -> str:
    source = re.sub(r'```mermaid\n.*?```', '\n**处理路径：** 图像或视频帧 → 检测与关键点 → 对齐 → 特征向量 → 余弦比较。检测与关键点也用于定位贴纸、美颜与三维重建。\n', source, flags=re.S)
    return markdown.markdown(source, extensions=['tables', 'fenced_code', 'toc', 'sane_lists'])


def make_html(source: Path, destination: Path, title: str):
    body = rendered('[TOC]\n\n'+source.read_text(encoding='utf-8'))
    body = re.sub(r'href="([a-z-]+)\.md([#"][^"]*)', r'href="\1.html\2', body)
    destination.write_text(f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title><style>{CSS}</style><body>{body}</body></html>', encoding='utf-8')


def source_files():
    result = []
    for folder in ['research','vision3d','scripts','tests','web','notebooks']:
        result.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and (p.suffix in {'.py','.html','.ipynb','.sh','.ps1','.yml','.yaml','.Dockerfile'} or p.name=='Dockerfile') and '__pycache__' not in p.parts)
    result.extend(p for p in ROOT.iterdir() if p.is_file() and (p.suffix in {'.py','.txt','.ps1','.toml','.yml'} or p.name in {'Dockerfile','.dockerignore','.gitignore'}))
    return sorted(set(result))


def use_of(path: Path) -> str:
    rel=path.relative_to(ROOT).as_posix()
    if rel=='app.py': return '本地网页服务；web/index.html 调用 /process 和 /verify。PDF任务2.4、4.3、9.2、9.3及系统集成。'
    if rel.startswith('web/'): return '由 app.py 的 GET / 返回；浏览器处理图像输入、效果控制、摄像头帧和双图验证。'
    if rel.startswith('vision3d/'): return '由 python -m vision3d.reconstruct 及其导出脚本调用；PDF任务8.2、8.3。'
    if rel.startswith('research/'): return '研究训练与评估命令使用；详见 research/README.md 和 docs/code-map.md。对应PDF任务3.2–7.3。'
    if rel.startswith('tests/'): return '回归检查；在项目根目录运行 python -m unittest discover -s tests。'
    if rel.startswith('notebooks/'): return 'Jupyter中的循序实验；PDF任务1.4、2.1、2.4和OpenCV入门。'
    if rel.startswith('scripts/'): return '模型下载、环境检查、数据准备或交付物生成；本文件开头docstring和下方函数索引提供具体入口。'
    if rel=='Dockerfile': return 'docker build --target hello 或 --target lab；PDF任务1.3。'
    if rel=='hello_world.py': return 'Docker hello镜像启动命令；PDF任务1.2/1.3。保留了旧仓库hello.py目录。'
    if rel=='start.ps1': return 'Windows本地启动与环境准备入口；项目根执行 powershell -File start.ps1。'
    return '依赖固定、打包或版本控制配置；由 pip、Docker 或 Git 读取。'


def make_compendium():
    files=source_files()
    chunks=['# 全部项目源码与使用位置\n', '此文档逐字收录本项目源文件，并提供使用位置、符号行号、原文件链接与SHA256。它由 `scripts/build_deliverables.py` 从实际文件自动生成，避免手工粘贴漏代码。第三方Python库与预训练权重不复制进本文；它们的版本、官方来源与许可证见环境锁定文件、`models/registry.json`、`models/licenses/`及`docs/sources.md`。\n', '## 如何阅读\n先查用途，再打开对应文件。文中的代码是完整源码，不是删节片段。Notebook收录全部代码单元，图像输出在原始ipynb中。算法的专业解释见主教程。\n', '## 文件目录\n\n| 文件 | 源码行数 | 使用位置 |\n|---|---:|---|']
    total=0
    for p in files:
        raw=p.read_text(encoding='utf-8')
        count=len(raw.splitlines()); total+=count
        rel=p.relative_to(ROOT).as_posix()
        chunks.append(f'| `{rel}` | {count} | {use_of(p)} |')
    chunks.append(f'\n共 {len(files)} 个源文件，{total} 行文本（Notebook按JSON行统计）。\n')
    for p in files:
        rel=p.relative_to(ROOT).as_posix(); raw=p.read_text(encoding='utf-8')
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        chunks.extend([f'\n## {rel}\n',f'**使用位置：** {use_of(p)}\n',f'[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/{rel})\n',f'SHA256：`{digest}`\n'])
        if p.suffix=='.py':
            try:
                tree=ast.parse(raw)
                symbols=[f'`{node.name}` 第{node.lineno}行' for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]
                if symbols: chunks.append('**代码定位：** '+'；'.join(symbols)+'。\n')
            except SyntaxError: pass
        if p.suffix=='.ipynb':
            cells=json.loads(raw)['cells']
            for index,cell in enumerate(cells,1):
                if cell['cell_type']=='code':
                    code=''.join(cell['source']) if isinstance(cell['source'],list) else cell['source']
                    chunks.append(f'### 第 {index} 个单元\n\n```python\n{code}\n```\n')
        else:
            language={'.py':'python','.html':'html','.ps1':'powershell','.sh':'bash','.yml':'yaml','.toml':'toml'}.get(p.suffix,'text')
            # Four backticks safely contain any triple fences embedded in a builder string.
            chunks.append(f'````{language}\n{raw}\n````\n')
    path=DOCS/'code-compendium.md';path.write_text('\n'.join(chunks),encoding='utf-8')
    make_html(path,DOCS/'code-compendium.html','全部源码与使用位置')
    (REPORTS/'code-inventory.json').write_text(json.dumps({'files':len(files),'lines':total,'paths':[p.relative_to(ROOT).as_posix() for p in files]},indent=2),encoding='utf-8')


def pdf_from_markdown(source: Path, destination: Path):
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if not font.exists(): raise FileNotFoundError('PDF generation requires Microsoft YaHei, or adjust font path')
    if 'Chinese' not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont('Chinese',str(font),subfontIndex=0))
        pdfmetrics.registerFontFamily('Chinese',normal='Chinese',bold='Chinese',italic='Chinese',boldItalic='Chinese')
    styles=getSampleStyleSheet()
    for style in styles.byName.values(): style.fontName='Chinese'
    body=ParagraphStyle('BodyCJK',fontName='Chinese',fontSize=10.2,leading=17,spaceAfter=8,wordWrap='CJK')
    code=ParagraphStyle('CodeCJK',fontName='Chinese',fontSize=7.2,leading=10.5,spaceAfter=10,backColor=colors.HexColor('#eef3ee'),borderPadding=7)
    heading=ParagraphStyle('HeadingCJK',parent=body,fontSize=16,leading=23,spaceBefore=20,spaceAfter=12,keepWithNext=True,textColor=colors.HexColor('#214b39'))
    small=ParagraphStyle('SmallCJK',parent=body,fontSize=8,leading=12)
    soup=BeautifulSoup(rendered(source.read_text(encoding='utf-8')),'html.parser')
    story=[]
    def inline(node):
        s=html.escape(node.get_text())
        return s.replace('\n','<br/>')
    for node in soup.children:
        if not getattr(node,'name',None): continue
        name=node.name
        if name in {'h1','h2','h3','h4'}:
            if name=='h2' and node.get_text().startswith('第') and story: story.append(PageBreak())
            style=heading if name!='h1' else ParagraphStyle('TitleCJK',parent=heading,fontSize=24,leading=34,spaceAfter=20)
            story.append(Paragraph(inline(node),style))
        elif name=='pre':
            lines=[]
            for line in node.get_text().splitlines():
                lines.extend(textwrap.wrap(line,width=84,expand_tabs=False,replace_whitespace=False,drop_whitespace=False) or [''])
            story.append(Preformatted('\n'.join(lines),code))
        elif name=='table':
            rows=[[Paragraph(inline(cell),small) for cell in row.find_all(['th','td'])] for row in node.find_all('tr')]
            if rows:
                table=Table(rows,colWidths=[(174*mm)/len(rows[0])]*len(rows[0]),repeatRows=1,hAlign='LEFT')
                table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5eee7')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#c8d7cc')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
                story.extend([table,Spacer(1,10)])
        elif name in {'ul','ol'}:
            for i,item in enumerate(node.find_all('li',recursive=False),1): story.append(Paragraph((f'{i}. ' if name=='ol' else '• ')+inline(item),body))
        elif name=='hr': story.append(Spacer(1,12))
        else: story.append(Paragraph(inline(node),body))
    def footer(canvas,doc):
        canvas.setFont('Chinese',8);canvas.setFillColor(colors.HexColor('#62776a'))
        canvas.drawString(18*mm,12*mm,'Face Vision Lab  ·  可复现实验与专业教程')
        canvas.drawRightString(192*mm,12*mm,str(doc.page))
    document=SimpleDocTemplate(str(destination),pagesize=(210*mm,297*mm),leftMargin=18*mm,rightMargin=18*mm,topMargin=18*mm,bottomMargin=22*mm,title=source.stem,author='Face Vision Lab')
    document.build(story,onFirstPage=footer,onLaterPages=footer)


def slides():
    metrics=json.loads((REPORTS/'demo-metrics.json').read_text(encoding='utf-8'))
    lfw_path=REPORTS/'lfw-sface-largest.json'
    lfw=json.loads(lfw_path.read_text(encoding='utf-8')) if lfw_path.exists() else json.loads((REPORTS/'lfw-sface.json').read_text(encoding='utf-8'))
    prs=Presentation();prs.slide_width=Inches(13.333);prs.slide_height=Inches(7.5)
    entries=[
      ('人脸视觉 从原理到可运行系统',['本地 CPU 应用 · 训练与评估管线 · 3D 重建','面向高三学生的可复现学习项目','独立教育项目，与字节跳动无隶属关系'],None),
      ('范围与验收口径',['PDF 含环境、检测、关键点、识别、优化、GAN、3D 与动态特效','预训练推理、代码冒烟测试、真实数据实验是三种不同证据','98.5% 是目标，必须由完整协议的实测报告支持'],None),
      ('系统如何处理一张图像',['图像 → YuNet 检测框与五点 → 对齐 → SFace 特征 → 余弦比较','关键点 → 几何变换 → 眼镜、皇冠、美颜、美妆','浏览器只连接本机；输入图像仅在内存处理'],None),
      ('真实运行界面',['六种显示模式；本机图片与摄像头路径','明确区分检测耗时与浏览器端到端延迟'], 'reports/app-screenshot.png'),
      ('环境与可重复性',['Windows 11 · Intel Core Ultra 5 225H · 31.5 GB RAM · CPU','Python 3.12.14 · PyTorch 2.14 CPU · OpenCV 5 · ONNX Runtime 1.30','独立环境、版本记录、模型哈希；Docker Hello World 已运行'],None),
      ('人脸识别与 ArcFace',['对齐后的人脸 → ResNet50 → L2 归一化特征','ArcFace 训练正类 logit：s cos(θ + m)，使类间更易区分','P-K 采样与 batch-hard triplet 辅助；实测反向传播及保存重读'],None),
      ('LFW 的真实验证结果',[f'完整 6,000 对 / 10 折；准确率 {lfw.get("accuracy_mean",0)*100:.2f}%','阈值仅由其余训练折确定；测试折不选阈值','预训练 SFace 基线；不能归属于本项目从零训练的 ResNet50'],None),
      ('一次失败怎样变成有效实验',['首轮严格单脸策略：72.25%；全部失败都来自多脸图','原图包含背景人脸；应用验证与数据集主体选择规则不同','保留失败报告，按最大主体规则独立重跑；不删除困难样本'], 'reports/lfw-detection-audit.jpg'),
      ('动态特效与性能',[f'演示视频 18 秒 / 360 帧；{metrics["frames_with_face"]} 帧检测到人脸',f'混合特效管线中位数 {metrics["pipeline_ms_median"]:.2f} ms，P95 {metrics["pipeline_ms_p95"]:.2f} ms','公开静态样本经仿射运动；不是摄像头实测或移动端性能'], 'reports/effect-all.jpg'),
      ('从单张照片重建三维',['3DDFA_V2：预测 62 参数，重建 38,365 顶点 / 76,073 三角面','导出 OBJ，多视角图，以及 WebGL 可旋转展示','单目统计估计，无真实尺度；没有 3D 真值误差测试'], 'reports/3d/multiview.png'),
      ('量化与 ONNX 不应只报好消息',['随机权重管线实测：ONNX 与 PyTorch 最大误差约 1.7×10⁻⁷','Linear 动态 int8：52.65 ms；FP32：48.84 ms，本轮没有加速','卷积占据 ResNet 主体；特征误差不等于识别准确率'],None),
      ('研究训练的完整路径',['WIDER → COCO 标注 → MMDetection；300-W → 关键点 → NME','身份文件夹 → ArcFace；CelebA 属性 → StarGAN → FID/IS','受数据授权、版本依赖与算力约束；逐项状态见验收矩阵'],None),
      ('版本与环境问题怎样解决',['缺系统目录变量 → 只补任务子进程；SSL/DNS/venv 恢复','Git LFS 指针 → 官方真实资产 URL + SHA256','旧 3D 代码 → 严格权重加载、ONNX + NumPy + WebGL'],None),
      ('专业学习路线与源码导航',['18 课：Python → 图像/向量 → 网络 → 评估 → 生成/3D → 部署','每课：准确概念、直观解释、运行步骤、练习与答案','全部源码汇编附用途、调用入口、函数行号和哈希'],None),
      ('开源交付与尚未完成的验收',['源码仓库：github.com/GBHLKYEric/face_ai_project','MIT 适用于本项目代码；权重与数据分别遵守原许可','完整训练达标、手机实测、受限数据/服务列入未完成清单'],None),
    ]
    for number,(title,bullets,picture) in enumerate(entries,1):
        slide=prs.slides.add_slide(prs.slide_layouts[6]);bg=slide.background.fill;bg.solid();bg.fore_color.rgb=RGBColor.from_string('F6F7F1')
        def textbox(x,y,w,h,text,size,color='20382D',bold=False):
            shape=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=shape.text_frame;tf.word_wrap=True
            p=tf.paragraphs[0];p.text=text;p.font.name='Microsoft YaHei';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=RGBColor.from_string(color)
            return shape
        textbox(.65,.35,11.7,.35,'FACE VISION LAB  /  PROJECT REVIEW',11,'647A69')
        textbox(.65,.95,12,1,title,30,bold=True)
        if picture and (ROOT/picture).exists():
            from PIL import Image
            with Image.open(ROOT/picture) as im: ratio=im.width/im.height
            boxw,boxh=6.2,4.7;w=min(boxw,boxh*ratio);h=w/ratio
            slide.shapes.add_picture(str(ROOT/picture),Inches(6.5+(boxw-w)/2),Inches(2.05+(boxh-h)/2),width=Inches(w),height=Inches(h))
            width=5.2
        else: width=11.6
        for i,bullet in enumerate(bullets):
            textbox(.7,2.25+i*1.25,width,1.15,bullet,22 if picture else 25)
        textbox(.65,7,10,.25,'可复现的代码、明确的证据、可理解的原理',10,'687E70')
        textbox(12,7,.6,.25,f'{number:02d}',10,'687E70')
    path=OUTPUT/'项目汇报.pptx';prs.save(path)
    check=Presentation(path)
    assert len(check.slides)==len(entries)
    for slide in check.slides:
        for shape in slide.shapes:
            assert shape.left>=0 and shape.top>=0 and shape.left+shape.width<=check.slide_width+10000 and shape.top+shape.height<=check.slide_height+10000
    print(f'Created {len(entries)} slides')


def main():
    make_compendium()
    for source in DOCS.glob('*.md'):
        if source.name=='code-compendium.md': continue
        dest=DOCS/('tutorial.html' if source.name=='tutorial-zh.md' else source.stem+'.html')
        make_html(source,dest,source.stem)
    pdf_from_markdown(DOCS/'tutorial-zh.md',OUTPUT/'高三零基础专业教程.pdf')
    for stem,name in [('project-report','项目总结报告.pdf'),('issues-and-fixes','执行问题与解决记录.pdf')]:
        if (DOCS/(stem+'.md')).exists(): pdf_from_markdown(DOCS/(stem+'.md'),OUTPUT/name)
    slides()
    print('Generated tutorial, reports, slides and complete code compendium.')


if __name__=='__main__': main()

````


## scripts/demo.py

**使用位置：** 模型下载、环境检查、数据准备或交付物生成；本文件开头docstring和下方函数索引提供具体入口。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/scripts/demo.py)

SHA256：`b43a20ee095686a2927c468623ad1801dbc5ff27000b114aa44d71b1909c99fa`

**代码定位：** `main` 第15行。

````python
"""Reproducible animated input + real per-frame inference, not a webcam recording."""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import Vision


def main():
    output = ROOT / 'reports'
    output.mkdir(exist_ok=True)
    sample = cv2.imread(str(ROOT / 'assets/sample.jpg'))
    if sample is None:
        raise FileNotFoundError('Run scripts/fetch_models.py first')
    vision = Vision()
    effects = ['none', 'glasses', 'crown', 'beauty', 'lipstick', 'all']
    fps, segment = 20, 60
    path = output / 'demo.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1024, 620))
    if not writer.isOpened():
        raise RuntimeError('MP4 writer unavailable')
    measurements, detected, samples = [], 0, []
    try:
        for index in range(segment * len(effects)):
            effect = effects[index // segment]
            transform = cv2.getRotationMatrix2D((256, 256), 9*np.sin(index/35), 1+.025*np.sin(index/24))
            transform[:, 2] += [12*np.sin(index/20), 5*np.sin(index/30)]
            frame = cv2.warpAffine(sample, transform, (512, 512), borderMode=cv2.BORDER_REFLECT)
            result, metrics = vision.process(frame, effect, .7, effect == 'none')
            measurements.append(metrics['pipeline_ms'])
            detected += metrics['faces'] > 0
            canvas = np.full((620, 1024, 3), (31, 42, 37), np.uint8)
            canvas[55:567, :512], canvas[55:567, 512:] = frame, result
            cv2.putText(canvas, 'FACE VISION LAB | CPU | '+effect.upper(), (28, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (224, 242, 220), 2)
            cv2.putText(canvas, 'Input: animated NASA public sample', (20, 590), cv2.FONT_HERSHEY_SIMPLEX, .55, (215, 225, 210), 1)
            cv2.putText(canvas, f'Real inference: {metrics["pipeline_ms"]:.1f} ms | faces {metrics["faces"]}', (534, 590), cv2.FONT_HERSHEY_SIMPLEX, .55, (215, 225, 210), 1)
            writer.write(canvas)
            if index % segment == segment // 2:
                image_path = output / f'effect-{effect}.jpg'
                cv2.imwrite(str(image_path), canvas)
                samples.append(str(image_path.name))
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    readable, _ = capture.read()
    capture.release()
    if frame_count != segment*len(effects) or not readable:
        raise RuntimeError('Video readback failed')
    result = {'video':path.name, 'source':'NASA astronaut public-domain sample; affine animated input, not a live camera',
              'frames':frame_count, 'playback_fps':fps, 'duration_seconds':frame_count/fps,
              'frames_with_face':detected, 'effects':effects, 'samples':samples,
              'pipeline_ms_median':float(np.median(measurements)), 'pipeline_ms_p95':float(np.percentile(measurements,95)),
              'pipeline_mean_fps':float(1000/np.mean(measurements)),
              'timing_scope':'YuNet detection + effects only; excludes decode/encode/browser display',
              'recorded_at':time.strftime('%Y-%m-%dT%H:%M:%S')}
    (output / 'demo-metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()

````


## scripts/doctor.py

**使用位置：** 模型下载、环境检查、数据准备或交付物生成；本文件开头docstring和下方函数索引提供具体入口。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/scripts/doctor.py)

SHA256：`0226a13041aeec5538ed901abc7a4f84d55dfa2bffcb9c8653d26c5502d0d68b`

**代码定位：** `report` 第17行。

````python
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

````


## scripts/fetch_models.py

**使用位置：** 模型下载、环境检查、数据准备或交付物生成；本文件开头docstring和下方函数索引提供具体入口。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/scripts/fetch_models.py)

SHA256：`f0e6846c0bb6be2772ada133d47e30f70b97d83cd510cdee72eabbf2116f4e65`

**代码定位：** `verify` 第11行；`fetch` 第15行。

````python
"""Download versioned, SHA256-verified assets from their official repositories."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def verify(path, expected):
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected


def fetch(with_3d=False):
    registry = json.loads((ROOT / 'models/registry.json').read_text(encoding='utf-8'))
    for item in registry['assets']:
        if item['group'] == '3d' and not with_3d:
            continue
        path = ROOT / item['path']
        if not verify(path, item['sha256']):
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + '.part')
            with urllib.request.urlopen(item['url'], timeout=180) as src, temporary.open('wb') as dst:
                while chunk := src.read(1024 * 1024):
                    dst.write(chunk)
            if not verify(temporary, item['sha256']):
                temporary.unlink(missing_ok=True)
                raise ValueError(f"SHA256 mismatch: {item['path']}")
            temporary.replace(path)
        print(f"Verified {item['path']}")
    import cv2
    img = cv2.imread(str(ROOT / 'assets/astronaut.png'))
    if img is None or not cv2.imwrite(str(ROOT / 'assets/sample.jpg'), img):
        raise RuntimeError('Could not create sample.jpg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-3d', action='store_true', help='Also download 3DDFA model assets; review models/README.md first.')
    fetch(parser.parse_args().with_3d)

````


## scripts/make_notebook.py

**使用位置：** 模型下载、环境检查、数据准备或交付物生成；本文件开头docstring和下方函数索引提供具体入口。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/scripts/make_notebook.py)

SHA256：`a4ab3c349247a1a0fa0262418db350042f9f47ce60a62d3f765e82b8a8c8e595`

````python
"""Build and execute the beginner notebook; outputs are preserved for inspection."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
cells = [
    nbf.v4.new_markdown_cell('# 从像素到人脸特效\n本笔记本用公开 NASA 示例完成真实推理。它不代表 LFW 测试或训练精度。按 Shift+Enter 逐格执行。'),
    nbf.v4.new_code_cell("from pathlib import Path\nimport sys\nimport cv2\nimport numpy as np\nimport matplotlib.pyplot as plt\nROOT = Path.cwd()\nif ROOT.name == 'notebooks': ROOT = ROOT.parent\nsys.path.insert(0, str(ROOT))\nfrom app import Vision\nprint('Python:', sys.version.split()[0], 'OpenCV:', cv2.__version__)"),
    nbf.v4.new_markdown_cell('## 1 图像就是数字数组\n彩色图像的形状是高×宽×3。OpenCV 的通道顺序是 BGR，显示前要换成 RGB。'),
    nbf.v4.new_code_cell("image = cv2.imread(str(ROOT / 'assets/sample.jpg'))\nassert image is not None\ngray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\nprint('彩色形状:', image.shape, '灰度形状:', gray.shape)\nfig, ax = plt.subplots(1, 2, figsize=(8,4))\nax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)); ax[1].imshow(gray, cmap='gray')\nfor a in ax: a.axis('off')\nplt.show()"),
    nbf.v4.new_markdown_cell('## 2 检测和五个关键点\n检测框给出位置，五个点给出姿态线索。看看转动图像以后，点是否仍跟随人脸。'),
    nbf.v4.new_code_cell("vision = Vision(ROOT / 'models')\nfaces = vision.detect(image)\nprint('检测到的人脸:', len(faces))\nprint('眼睛、鼻尖、嘴角:', faces[0,4:14].reshape(5,2))\nresult, metrics = vision.process(image, 'all', .6, True)\nplt.figure(figsize=(5,5)); plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)); plt.axis('off'); plt.show()\nprint(metrics)"),
    nbf.v4.new_markdown_cell('## 3 向量的余弦\n点积除以两向量长度。相同方向为 1，垂直为 0，反向为 −1。'),
    nbf.v4.new_code_cell("def cosine(a,b):\n    a,b=np.asarray(a,float),np.asarray(b,float)\n    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))\nprint(cosine([1,0],[1,0]), cosine([1,0],[0,1]), cosine([1,0],[-1,0]))\nassert cosine([1,0],[0,1]) == 0\nprint(vision.verify(image,image,.363))"),
    nbf.v4.new_markdown_cell('## 4 动手练习\n1. 把特效改为 glasses、crown 或 lipstick。\n2. 将图像旋转 10 度，再比较关键点。\n3. 为什么一对相同图像通过验证，不能说明模型准确率是 100%？\n\n答案：没有负样本、姿态/光照变化、独立测试集，也没有足够样本。LFW 实验必须执行 research.lfw 的完整协议。'),
]
notebook = nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
folder = ROOT / 'notebooks'
folder.mkdir(exist_ok=True)
path = folder / '01-first-vision-lab.ipynb'
nbf.write(notebook, path)
NotebookClient(notebook, timeout=180, kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
nbf.write(notebook, path)
print(f'Executed {sum(c.cell_type == "code" for c in notebook.cells)} code cells: {path.name}')

````


## scripts/mmdet.Dockerfile

**使用位置：** 模型下载、环境检查、数据准备或交付物生成；本文件开头docstring和下方函数索引提供具体入口。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/scripts/mmdet.Dockerfile)

SHA256：`f7059cfe2c76d6e453e0684ac0b54390af41bf192e493b9bcc9341b9feb64240`

````text
FROM python:3.10-slim-bookworm
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pip==25.3 'setuptools<81'
RUN pip install --no-cache-dir numpy==1.26.4 torch==2.1.0+cpu torchvision==0.16.0+cpu --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/mmcv-2.1.0-cp310-cp310-manylinux1_x86_64.whl mmengine==0.10.7 mmdet==3.3.0 'opencv-python<4.12' 'yapf==0.40.1'
RUN python -c "import torch,mmcv,mmdet;from mmcv.ops import nms;print(torch.__version__,mmcv.__version__,mmdet.__version__);print(nms(torch.tensor([[0.,0.,10.,10.],[1.,1.,11.,11.]]),torch.tensor([.9,.8]),.5))"
WORKDIR /project
CMD ["python", "scripts/mmdet_pilot.py"]

````


## scripts/mmdet_pilot.py

**使用位置：** 模型下载、环境检查、数据准备或交付物生成；本文件开头docstring和下方函数索引提供具体入口。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/scripts/mmdet_pilot.py)

SHA256：`40c34cb17ae2d0d9efe7431c8c6ace3c355ff1e445c4a0340462f9552f0286ea`

**代码定位：** `main` 第19行。

````python
"""Train and evaluate a small real WIDER subset in the separate MMDet container.

This deliberately short CPU experiment validates the training/evaluation path.
It is neither a converged detector nor the official WIDER easy/medium/hard AP.
"""
import json
from pathlib import Path
import time

import torch
import mmcv
import mmdet
import mmengine
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.utils import register_all_modules


def main():
    root = Path.cwd()
    for split in ['train', 'val']:
        if not (root / f'data/wider/{split}.json').is_file():
            raise FileNotFoundError('First run python -m research.fetch_wider_subset')
    torch.set_num_threads(4)
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile('research/configs/wider_retinanet.py')
    cfg.model.backbone.init_cfg = None  # no unreported ImageNet pretrained weights
    # Keep ranked low-confidence candidates so the evaluator can report actual AP.
    # A scratch model's ~0.01 scores otherwise all disappear at the default .05.
    cfg.model.test_cfg.score_thr = 0.0
    cfg.train_dataloader.batch_size = 2
    cfg.train_dataloader.num_workers = 0
    cfg.train_dataloader.persistent_workers = False
    cfg.val_dataloader.num_workers = 0
    cfg.val_dataloader.persistent_workers = False
    cfg.test_dataloader = cfg.val_dataloader
    cfg.train_dataloader.dataset.pipeline = [
        dict(type='LoadImageFromFile'), dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', scale=(320, 320), keep_ratio=True), dict(type='RandomFlip', prob=.5),
        dict(type='PackDetInputs')]
    cfg.val_dataloader.dataset.pipeline = [
        dict(type='LoadImageFromFile'), dict(type='Resize', scale=(320, 320), keep_ratio=True),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='PackDetInputs', meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor'))]
    cfg.train_cfg.max_epochs = 1
    cfg.train_cfg.val_interval = 2  # evaluate explicitly once after the one training epoch
    cfg.param_scheduler = []
    cfg.auto_scale_lr = dict(enable=False, base_batch_size=2)
    cfg.env_cfg.dist_cfg.backend = 'gloo'
    cfg.randomness = dict(seed=42, deterministic=False)
    cfg.default_hooks.logger.interval = 1
    cfg.default_hooks.checkpoint.interval = 1
    cfg.work_dir = 'runs/wider_pilot'
    start = time.perf_counter()
    runner = Runner.from_cfg(cfg)
    runner.train()
    metrics = runner.val()
    from mmdet.apis import inference_detector
    import cv2
    runner.model.cfg = cfg
    validation = json.loads((root/'data/wider/val.json').read_text())
    sample_path = root/'data/WIDER_val/images'/validation['images'][0]['file_name']
    prediction = inference_detector(runner.model, str(sample_path)).pred_instances.cpu()
    canvas = cv2.imread(str(sample_path))
    candidates = prediction.scores.argsort(descending=True)[:10]
    for index in candidates:
        x1, y1, x2, y2 = prediction.bboxes[index].numpy().astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 180, 255), 2)
        cv2.putText(canvas, f'{float(prediction.scores[index]):.3f}', (x1, max(15, y1)),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 180, 255), 1)
    Path('reports').mkdir(exist_ok=True)
    cv2.imwrite('reports/wider_pilot_prediction.jpg', canvas)
    counts = {split: len(json.loads((root/f'data/wider/{split}.json').read_text())['images'])
              for split in ['train', 'val']}
    report = {'purpose': 'CPU training/inference/evaluation pipeline pilot on real WIDER subset',
              'dataset_images': counts, 'epochs': 1, 'seed': 42, 'input_resize': [320, 320],
              'initialization': 'random, no pretrained backbone', 'model': 'RetinaNet ResNet50 FPN, one face class',
              'evaluation_score_threshold': 0.0,
              'wall_seconds': time.perf_counter()-start, 'metrics': metrics,
              'metric_protocol': 'COCO AP 0.50:0.95 on selected validation images; NOT official WIDER AP',
              'versions': {'torch': torch.__version__, 'mmcv': mmcv.__version__,
                           'mmengine': mmengine.__version__, 'mmdet': mmdet.__version__},
              'inference_visualization': 'Top 10 candidate boxes, including low scores, from un-converged pilot model',
              'limitations': '32/16 images and one epoch do not establish generalization or convergence.'}
    Path('reports').mkdir(exist_ok=True)
    Path('reports/wider_pilot.json').write_text(json.dumps(report, indent=2, default=float), encoding='utf-8')
    print(json.dumps(report, indent=2, default=float))


if __name__ == '__main__':
    main()

````


## scripts/verify_local.py

**使用位置：** 模型下载、环境检查、数据准备或交付物生成；本文件开头docstring和下方函数索引提供具体入口。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/scripts/verify_local.py)

SHA256：`1460f9296e11c6f234a66cf0df3f70f2b12287b031dbfcd457b500d37405031b`

**代码定位：** `request` 第12行；`main` 第21行。

````python
"""HTTP smoke and boundary check of the running local app, without a browser."""
import base64
import json
from pathlib import Path
import urllib.request
import urllib.error

ROOT=Path(__file__).resolve().parents[1]
BASE='http://127.0.0.1:8765'


def request(path, value=None, extra=None):
    data=json.dumps(value).encode() if value is not None else None
    headers={'Content-Type':'application/json',**(extra or {})}
    req=urllib.request.Request(BASE+path,data=data,headers=headers)
    try:
        with urllib.request.urlopen(req,timeout=30) as r: return r.status,r.read(),r.headers.get('Content-Type')
    except urllib.error.HTTPError as e: return e.code,e.read(),e.headers.get('Content-Type')


def main():
    checks=[]
    for path in ['/','/health','/tutorial','/3d','/sample.jpg']:
        status,body,mime=request(path);assert status==200,(path,status)
        checks.append({'path':path,'status':status,'bytes':len(body),'content_type':mime})
    assert request('/not-here')[0]==404
    image='data:image/jpeg;base64,'+base64.b64encode((ROOT/'assets/sample.jpg').read_bytes()).decode()
    status,body,_=request('/process',{'image':image,'effect':'all','strength':.7,'landmarks':True})
    result=json.loads(body);assert status==200 and result['faces']==1 and result['image'].startswith('data:image/jpeg;base64,')
    checks.append({'path':'/process','status':status,'faces':result['faces'],'pipeline_ms':result['pipeline_ms']})
    status,body,_=request('/verify',{'first':image,'second':image,'threshold':.363})
    result=json.loads(body);assert status==200 and result['match'] and abs(result['cosine']-1)<1e-5
    checks.append({'path':'/verify','status':status,'same_image_cosine':result['cosine']})
    for value in [{'image':'invalid'},{'image':image,'effect':'invalid'},{'image':image,'strength':3},[]]:
        assert request('/process',value)[0]==400
    assert request('/process',{'image':image},{'Origin':'https://untrusted.example'})[0]==403
    checks.append({'invalid_inputs':'400','cross_origin':'403','unknown_path':'404'})
    (ROOT/'reports/http-verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))


if __name__=='__main__': main()

````


## start.ps1

**使用位置：** Windows本地启动与环境准备入口；项目根执行 powershell -File start.ps1。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/start.ps1)

SHA256：`ddbb7a7bd71d84f6eae71d45e17c1ed556723631a9011fbfbda597039e46e6ed`

````powershell
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
# Repair restricted child-process environments without changing system settings.
if (-not $env:SystemRoot) { $env:SystemRoot = 'C:\Windows' }
if (-not $env:WINDIR) { $env:WINDIR = $env:SystemRoot }
if (-not $env:COMSPEC) { $env:COMSPEC = "$env:SystemRoot\System32\cmd.exe" }
$env:PYTHONIOENCODING = 'utf-8'
if (-not (Test-Path '.venv/Scripts/python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python virtual environment creation failed.' }
}
& .venv/Scripts/python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& .venv/Scripts/python.exe scripts/fetch_models.py
if ($LASTEXITCODE -ne 0) { throw 'Model download failed.' }
Start-Process 'http://127.0.0.1:8765'
& .venv/Scripts/python.exe app.py

````


## tests/test_app.py

**使用位置：** 回归检查；在项目根目录运行 python -m unittest discover -s tests。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/tests/test_app.py)

SHA256：`b42c083da89e326e681561da8a25038d3f057e45c469b8119409172d7d9b743b`

**代码定位：** `AppCheck` 第13行。

````python
"""One compact check of the real image pipeline and validation boundaries."""
import sys
from pathlib import Path
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import Vision, decode_image, encode_image, EFFECTS


class AppCheck(unittest.TestCase):
    def test_pipeline_and_boundaries(self):
        root = Path(__file__).resolve().parents[1]
        vision = Vision(root / 'models')
        sample = cv2.imread(str(root / 'assets/sample.jpg'))
        self.assertIsNotNone(sample)
        self.assertEqual(len(vision.detect(sample)), 1)
        self.assertEqual(decode_image(encode_image(sample)).shape, sample.shape)
        for effect in EFFECTS:
            out, metrics = vision.process(sample, effect, .5, False)
            self.assertEqual(out.shape, sample.shape)
            self.assertEqual(out.dtype, np.uint8)
            self.assertEqual(metrics['faces'], 1)
            self.assertGreater(metrics['pipeline_ms'], 0)
            if effect != 'none':
                self.assertTrue(np.any(out != sample))
        self.assertAlmostEqual(vision.verify(sample, sample, .363)['cosine'], 1, places=5)
        blank = np.zeros_like(sample)
        self.assertEqual(len(vision.detect(blank)), 0)
        with self.assertRaises(ValueError):
            vision.feature(blank)
        for malformed in [None, '', 'data:image/jpeg;base64,!', 'https://example.com/a.jpg']:
            with self.assertRaises(ValueError):
                decode_image(malformed)
        with self.assertRaises(ValueError):
            vision.process(sample, 'unknown', .5, False)


if __name__ == '__main__':
    unittest.main()

````


## tests/test_research.py

**使用位置：** 回归检查；在项目根目录运行 python -m unittest discover -s tests。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/tests/test_research.py)

SHA256：`c66f75f7714390bc22030d6300b479e79d56f3630526f2787b440a9e0d19b328`

**代码定位：** `ResearchChecks` 第13行。

````python
"""Small regression check for parsers, held-out thresholds and NME mathematics."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from research.lfw import evaluate_scores,parse_pairs
from research.landmarks import nme
from research.data import wider_to_coco


class ResearchChecks(unittest.TestCase):
    def test_protocol_and_geometry(self):
        # Deliberately reversed second fold: held-out threshold cannot cheat.
        report=evaluate_scores([.9,.1,.1,.9],[1,0,1,0],[0,0,1,1])
        self.assertEqual(report['accuracy_mean'],.25)
        target=np.zeros((1,68,2)); target[:,45,0]=10
        self.assertEqual(float(nme(target,target)[0]),0.)
        self.assertAlmostEqual(float(nme(target+1,target)[0]),np.sqrt(2)/10)
        with self.assertRaises(ValueError): nme(target,np.zeros_like(target))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ['Alice','Bob']:
                (root/name).mkdir()
                for number in [1,2]: Image.new('RGB',(20,20)).save(root/name/f'{name}_{number:04d}.jpg')
            pairs=root/'pairs.txt'; pairs.write_text('2 1\nAlice 1 2\nAlice 1 Bob 1\nBob 1 2\nBob 1 Alice 1\n')
            self.assertEqual(len(parse_pairs(root,pairs,strict=False)),4)
            with self.assertRaises(ValueError): parse_pairs(root,pairs)
            pairs.write_text('2 1\n../Alice 1 2\nAlice 1 Bob 1\nBob 1 2\nBob 1 Alice 1\n')
            with self.assertRaises(ValueError): parse_pairs(root,pairs,strict=False)
            annotation=root/'wider.txt'
            annotation.write_text('Alice/Alice_0001.jpg\n2\n-2 -2 10 10 0 0 0 0 0 0\n1 1 3 3 0 0 0 1 0 0\nBob/Bob_0001.jpg\n0\n0 0 0 0 0 0 0 0 0 0\n')
            report=wider_to_coco(root,annotation,root/'coco.json')
            self.assertEqual(report,{'images':2,'valid_boxes':1,'excluded_boxes':1})
            self.assertEqual(json.loads((root/'coco.json').read_text())['annotations'][0]['bbox'],[0,0,8,8])


if __name__=='__main__': unittest.main()

````


## vision3d/__init__.py

**使用位置：** 由 python -m vision3d.reconstruct 及其导出脚本调用；PDF任务8.2、8.3。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/vision3d/__init__.py)

SHA256：`dbd0c19ebc3cde758fd5d0bdb9f73ebf7edd6486e7af4d53bd6116b13fa035c5`

````python
"""3DDFA_V2 single-image 3D morphable face reconstruction."""

````


## vision3d/check.py

**使用位置：** 由 python -m vision3d.reconstruct 及其导出脚本调用；PDF任务8.2、8.3。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/vision3d/check.py)

SHA256：`537c2c5472e204ba8d2563a5c574f5de622ebbe6c76f022412323eae39f0dd3d`

**代码定位：** `main` 第10行。

````python
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

````


## vision3d/mobilenet_v1.py

**使用位置：** 由 python -m vision3d.reconstruct 及其导出脚本调用；PDF任务8.2、8.3。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/vision3d/mobilenet_v1.py)

SHA256：`b05dcb4349b2be5b0e9cb479a6c8baf7e8626f4f8aca3386aafbeeef40354cbb`

**代码定位：** `DepthWiseBlock` 第22行；`MobileNet` 第48行；`mobilenet` 第122行；`mobilenet_2` 第141行；`mobilenet_1` 第146行；`mobilenet_075` 第151行；`mobilenet_05` 第156行；`mobilenet_025` 第161行。

````python
# coding: utf-8

from __future__ import division

""" 
Creates a MobileNet Model as defined in:
Andrew G. Howard Menglong Zhu Bo Chen, et.al. (2017). 
MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. 
Copyright (c) Yang Lu, 2017

Modified By cleardusk
"""
import math
import torch.nn as nn

__all__ = ['MobileNet', 'mobilenet']


# __all__ = ['mobilenet_2', 'mobilenet_1', 'mobilenet_075', 'mobilenet_05', 'mobilenet_025']


class DepthWiseBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, prelu=False):
        super(DepthWiseBlock, self).__init__()
        inplanes, planes = int(inplanes), int(planes)
        self.conv_dw = nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, stride=stride, groups=inplanes,
                                 bias=False)
        self.bn_dw = nn.BatchNorm2d(inplanes)
        self.conv_sep = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_sep = nn.BatchNorm2d(planes)
        if prelu:
            self.relu = nn.PReLU()
        else:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv_dw(x)
        out = self.bn_dw(out)
        out = self.relu(out)

        out = self.conv_sep(out)
        out = self.bn_sep(out)
        out = self.relu(out)

        return out


class MobileNet(nn.Module):
    def __init__(self, widen_factor=1.0, num_classes=1000, prelu=False, input_channel=3):
        """ Constructor
        Args:
            widen_factor: config of widen_factor
            num_classes: number of classes
        """
        super(MobileNet, self).__init__()

        block = DepthWiseBlock
        self.conv1 = nn.Conv2d(input_channel, int(32 * widen_factor), kernel_size=3, stride=2, padding=1,
                               bias=False)

        self.bn1 = nn.BatchNorm2d(int(32 * widen_factor))
        if prelu:
            self.relu = nn.PReLU()
        else:
            self.relu = nn.ReLU(inplace=True)

        self.dw2_1 = block(32 * widen_factor, 64 * widen_factor, prelu=prelu)
        self.dw2_2 = block(64 * widen_factor, 128 * widen_factor, stride=2, prelu=prelu)

        self.dw3_1 = block(128 * widen_factor, 128 * widen_factor, prelu=prelu)
        self.dw3_2 = block(128 * widen_factor, 256 * widen_factor, stride=2, prelu=prelu)

        self.dw4_1 = block(256 * widen_factor, 256 * widen_factor, prelu=prelu)
        self.dw4_2 = block(256 * widen_factor, 512 * widen_factor, stride=2, prelu=prelu)

        self.dw5_1 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_2 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_3 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_4 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_5 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_6 = block(512 * widen_factor, 1024 * widen_factor, stride=2, prelu=prelu)

        self.dw6 = block(1024 * widen_factor, 1024 * widen_factor, prelu=prelu)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(int(1024 * widen_factor), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.dw2_1(x)
        x = self.dw2_2(x)
        x = self.dw3_1(x)
        x = self.dw3_2(x)
        x = self.dw4_1(x)
        x = self.dw4_2(x)
        x = self.dw5_1(x)
        x = self.dw5_2(x)
        x = self.dw5_3(x)
        x = self.dw5_4(x)
        x = self.dw5_5(x)
        x = self.dw5_6(x)
        x = self.dw6(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def mobilenet(**kwargs):
    """
    Construct MobileNet.
    widen_factor=1.0  for mobilenet_1
    widen_factor=0.75 for mobilenet_075
    widen_factor=0.5  for mobilenet_05
    widen_factor=0.25 for mobilenet_025
    """
    # widen_factor = 1.0, num_classes = 1000
    # model = MobileNet(widen_factor=widen_factor, num_classes=num_classes)
    # return model

    model = MobileNet(
        widen_factor=kwargs.get('widen_factor', 1.0),
        num_classes=kwargs.get('num_classes', 62)
    )
    return model


def mobilenet_2(num_classes=62, input_channel=3):
    model = MobileNet(widen_factor=2.0, num_classes=num_classes, input_channel=input_channel)
    return model


def mobilenet_1(num_classes=62, input_channel=3):
    model = MobileNet(widen_factor=1.0, num_classes=num_classes, input_channel=input_channel)
    return model


def mobilenet_075(num_classes=62, input_channel=3):
    model = MobileNet(widen_factor=0.75, num_classes=num_classes, input_channel=input_channel)
    return model


def mobilenet_05(num_classes=62, input_channel=3):
    model = MobileNet(widen_factor=0.5, num_classes=num_classes, input_channel=input_channel)
    return model


def mobilenet_025(num_classes=62, input_channel=3):
    model = MobileNet(widen_factor=0.25, num_classes=num_classes, input_channel=input_channel)
    return model

````


## vision3d/reconstruct.py

**使用位置：** 由 python -m vision3d.reconstruct 及其导出脚本调用；PDF任务8.2、8.3。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/vision3d/reconstruct.py)

SHA256：`1276c26c98e90a5345ae3790ec6c14ba86823361de44341cca3f2ad5300ece86`

**代码定位：** `NumpyOnlyUnpickler` 第21行；`checked_asset` 第35行；`load_array_pickle` 第46行；`export_onnx` 第51行；`reconstruct` 第67行；`save_result` 第111行；`main` 第164行。

````python
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

````


## vision3d/viewer-template.html

**使用位置：** 由 python -m vision3d.reconstruct 及其导出脚本调用；PDF任务8.2、8.3。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/vision3d/viewer-template.html)

SHA256：`82b9a69364de1bee25d0679c1756b39239e659b90f40b1e91e6deaa0d1428e02`

````html
<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>3DDFA 人脸三维重建 · WebGL</title>
<style>body{margin:0;background:#101726;color:#edf5f7;font:17px system-ui}main{max-width:1100px;margin:32px auto;padding:0 24px}h1{font-size:30px}p{color:#bbc6d6;line-height:1.7}canvas{width:100%;height:65vh;min-height:360px;background:#131e31;border-radius:16px}label{display:flex;align-items:center;gap:20px;margin:18px 0}input{flex:1}a{color:#6edddd}</style>
<main><h1>一张照片估计出的三维人脸</h1><p>3DDFA_V2 预测 62 个姿态、形状和表情参数，再由 3DMM 生成 38,365 个顶点。拖动滑块观察不同角度。此模型没有真实尺度，也不是三维扫描结果。</p>
<canvas id="mesh" aria-label="可旋转的三维人脸模型"></canvas><label for="yaw">左右旋转<input id="yaw" type="range" min="-90" max="90" value="0"><output id="angle">0°</output></label><p id="status"></p>
<p><a href="face.obj">下载 OBJ 模型</a> · <a href="multiview.png">查看三个角度的静态图</a></p></main>
<script>
const data=__MESH_DATA__;
const canvas=document.getElementById('mesh'), status=document.getElementById('status'), gl=canvas.getContext('webgl');
if(!gl){status.textContent='浏览器未启用 WebGL，请查看静态图。';}else{
 const vertex=`attribute vec3 p;attribute vec3 n;uniform float angle;uniform float aspect;varying float light;void main(){float c=cos(angle),s=sin(angle);mat3 r=mat3(c,0.,s,0.,1.,0.,-s,0.,c);vec3 q=r*p;vec3 normal=normalize(r*n);light=.30+.70*abs(dot(normal,normalize(vec3(.4,.5,1.))));gl_Position=vec4(q.x*1.7/aspect,q.y*1.7,q.z*.5,1.);}`;
 const fragment=`precision mediump float;varying float light;void main(){gl_FragColor=vec4(vec3(.23,.79,.83)*light,1.);}`;
 function shader(type,text){const v=gl.createShader(type);gl.shaderSource(v,text);gl.compileShader(v);if(!gl.getShaderParameter(v,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(v));return v;}
 const prog=gl.createProgram();gl.attachShader(prog,shader(gl.VERTEX_SHADER,vertex));gl.attachShader(prog,shader(gl.FRAGMENT_SHADER,fragment));gl.linkProgram(prog);if(!gl.getProgramParameter(prog,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(prog));gl.useProgram(prog);
 function attribute(name,values){gl.bindBuffer(gl.ARRAY_BUFFER,gl.createBuffer());gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(values),gl.STATIC_DRAW);const loc=gl.getAttribLocation(prog,name);gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,3,gl.FLOAT,false,0,0);}
 attribute('p',data.vertices);attribute('n',data.normals);gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,gl.createBuffer());gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,new Uint16Array(data.triangles),gl.STATIC_DRAW);gl.enable(gl.DEPTH_TEST);gl.clearColor(.075,.118,.192,1);
 function draw(){const ratio=devicePixelRatio||1;canvas.width=Math.round(canvas.clientWidth*ratio);canvas.height=Math.round(canvas.clientHeight*ratio);gl.viewport(0,0,canvas.width,canvas.height);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.uniform1f(gl.getUniformLocation(prog,'angle'),Number(yaw.value)*Math.PI/180);gl.uniform1f(gl.getUniformLocation(prog,'aspect'),canvas.width/canvas.height);gl.drawElements(gl.TRIANGLES,data.triangles.length,gl.UNSIGNED_SHORT,0);document.getElementById('angle').textContent=yaw.value+'°';}
 const yaw=document.getElementById('yaw');yaw.addEventListener('input',draw);window.addEventListener('resize',draw);draw();status.textContent='WebGL（基于 OpenGL ES 2.0）已渲染 · '+data.vertices.length/3+' 个顶点 · '+data.triangles.length/3+' 个三角面';
}
</script></html>

````


## web/index.html

**使用位置：** 由 app.py 的 GET / 返回；浏览器处理图像输入、效果控制、摄像头帧和双图验证。

[GitHub中的文件](https://github.com/GBHLKYEric/face_ai_project/blob/main/web/index.html)

SHA256：`9c543e3ffedeb3a70a5929757db12576c61cf8cf89b141159d02493355933f5b`

````html
<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Face Vision Lab · 智能视觉实验室</title>
<style>
:root{font-family:"Segoe UI","Microsoft YaHei",sans-serif;color:#19242b;background:#f5f5f0;line-height:1.6}*{box-sizing:border-box}body{margin:0}header{border-bottom:1px solid #d9dfd9;padding:20px 5vw;display:flex;justify-content:space-between;gap:20px;align-items:center}header strong{letter-spacing:2px}header a,a{color:#23634e}main{max-width:1320px;margin:auto;padding:35px 5vw}h1{font-size:clamp(28px,4vw,46px);margin:6px 0 12px;letter-spacing:-1px;font-weight:650}p{margin:8px 0}.eyebrow{font-size:12px;color:#3f6a57;letter-spacing:3px}.intro{max-width:750px;color:#52635a;margin-bottom:28px}.workspace{display:grid;grid-template-columns:minmax(0,1fr) 280px;gap:26px}.stage{position:relative;border-radius:16px;background:#142b28;min-height:420px;display:flex;align-items:center;justify-content:center;overflow:hidden}#result{display:block;max-width:100%;max-height:610px;object-fit:contain}.stage-label{position:absolute;left:18px;top:14px;background:#ffffffdf;border-radius:20px;padding:3px 12px;font-size:12px}.controls{background:#fff;border:1px solid #dce2dc;border-radius:16px;padding:22px}h2{font-size:18px;margin:0 0 16px}button,select,input{font:inherit}button,.file{display:inline-block;border:1px solid #b3c4b8;border-radius:8px;background:#fff;color:#243e32;padding:9px 14px;cursor:pointer}button:hover,.file:hover{background:#e8eee7}button.active,button.primary{background:#214d3b;color:white;border-color:#214d3b}button:disabled{opacity:.5;cursor:wait}button:focus-visible,input:focus-visible,a:focus-visible{outline:3px solid #b09a42;outline-offset:3px}.buttons{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0}.effects{display:grid;grid-template-columns:1fr 1fr;gap:8px}.effects button{padding:10px 5px}label{display:block;margin-top:20px;font-size:14px}input[type=range]{width:100%;accent-color:#295c46}input[type=file]{max-width:100%;font-size:13px}.muted{font-size:13px;color:#5e7167}.stats{display:flex;flex-wrap:wrap;gap:26px;padding:17px 4px}.stat{color:#5e7167;font-size:12px}.stat b{display:block;color:#243e32;font-size:24px;font-weight:500}.divider{height:1px;background:#dce2dc;margin:24px 0}details{border-top:1px solid #ced9cf;padding:22px 0;margin-top:14px}summary{font-size:19px;cursor:pointer}.compare{display:flex;flex-wrap:wrap;align-items:end;gap:20px}.compare label{max-width:280px}.notice{padding:12px 0;min-height:48px;color:#405b4c}#error{color:#a52c37;min-height:26px}.learning{display:grid;grid-template-columns:repeat(3,1fr);gap:24px;margin:32px 0}.learning h3{font-size:16px}.learning p{font-size:14px;color:#52635a}footer{color:#637168;font-size:12px;padding:28px 0;border-top:1px solid #d9dfd9}video,#capture{display:none}@media(max-width:800px){.workspace{grid-template-columns:1fr}.stage{min-height:300px}.effects{grid-template-columns:repeat(3,1fr)}.learning{grid-template-columns:1fr}header{align-items:start;font-size:13px}main{padding-top:24px}}
</style>
<header><strong>FACE VISION LAB</strong><span>本机处理 · 开源学习项目　<a href="/3d" target="_blank">3D 重建 ↗</a>　<a href="/tutorial" target="_blank">学习教程 ↗</a></span></header>
<main>
<div class="eyebrow">COMPUTER VISION / LEARN BY MAKING</div>
<h1>从一张脸，理解智能视觉。</h1>
<p class="intro">观察检测框与五个关键点，试试随人脸位置变化的特效，再比较两张照片的特征相似度。图像仅在本机内存处理，不上传到外部服务器。</p>
<div class="workspace">
<div>
<div class="stage"><span class="stage-label" id="source-label">公开示例 · NASA 宇航员照片</span><img id="result" alt="人脸检测和特效处理结果" src="/sample.jpg"></div>
<div class="stats"><div class="stat">检测到的人脸<b id="faces">—</b></div><div class="stat">检测 + 特效耗时<b id="latency">—</b></div><div class="stat">处理能力估计<b id="fps">—</b></div></div>
<p class="muted">耗时为 CPU 检测与特效处理，未计传输和显示；完整往返耗时：<span id="roundtrip">—</span>。画面逐帧检测驱动贴纸。</p>
</div>
<aside class="controls"><h2>视觉实验台</h2>
<div class="effects" id="effects"><button class="active" data-effect="none" aria-pressed="true">原始检测</button><button data-effect="glasses" aria-pressed="false">AR 眼镜</button><button data-effect="crown" aria-pressed="false">皇冠贴纸</button><button data-effect="beauty" aria-pressed="false">柔肤提亮</button><button data-effect="lipstick" aria-pressed="false">几何口红</button><button data-effect="all" aria-pressed="false">组合特效</button></div>
<label for="strength">特效强度 <output id="amount">50%</output></label><input id="strength" type="range" min="0" max="100" value="50">
<label><input id="landmarks" type="checkbox" checked> 显示检测框与关键点</label>
<div class="divider"></div>
<label for="upload">选择本机图像</label><input id="upload" type="file" accept="image/png,image/jpeg,image/webp">
<div class="buttons"><button id="camera" class="primary">开启摄像头</button><button id="stop" hidden>停止</button><button id="sample">恢复示例</button><button id="save">保存结果</button></div>
<p class="muted">摄像头由你授权后开启；停止即释放。口红使用五点几何近似，不是唇部分割。此项目与字节跳动无隶属关系。</p></aside>
</div>
<p id="error" role="alert"></p>
<details><summary>双图人脸验证 · 相似不等于确定身份</summary><p class="muted">分别选择恰好包含一张脸的两张图像。SFace 比较归一化特征，阈值需要在你的独立验证集校准。</p>
<div class="compare"><label>图像 A<input id="first" type="file" accept="image/*"></label><label>图像 B<input id="second" type="file" accept="image/*"></label><label>余弦阈值<input id="threshold" type="number" value="0.363" min="-1" max="1" step="0.01"></label><button id="verify">比较两张图像</button></div><p id="comparison" class="notice" aria-live="polite">等待选择图像。</p></details>
<div class="learning"><section><h3>01 / 找到人脸</h3><p>YuNet 输出人脸框、双眼、鼻尖和两个嘴角。点的位置决定贴纸的平移、缩放与旋转。</p></section><section><h3>02 / 提取特征</h3><p>对齐后的脸进入 SFace 网络，变为特征向量。余弦相似度衡量方向接近程度。</p></section><section><h3>03 / 验证证据</h3><p>界面演示、管线测试和数据集准确率是不同证据。完整训练流程与待完成实验详见教程。</p></section></div>
<footer>开源教育实验 · CPU 本地推理 · 不保存人脸库 · 模型与数据许可见项目文档</footer>
</main><video id="video" playsinline muted></video><canvas id="capture"></canvas>
<script>
const $=id=>document.getElementById(id);let source=null,effect='none',stream=null,busy=false,pending=false,cameraGeneration=0;
const read=file=>new Promise((resolve,reject)=>{if(!file)return reject(Error('请先选择图像'));if(file.size>8000000)return reject(Error('图像不得超过 8 MB'));let r=new FileReader;r.onload=()=>resolve(r.result);r.onerror=()=>reject(Error('读取失败'));r.readAsDataURL(file)});
async function request(path,data){let r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});let j=await r.json();if(!r.ok)throw Error(j.error||'处理失败');return j}
async function process(){if(!source)return;if(busy){pending=true;return}busy=true;let t=performance.now();try{let j=await request('/process',{image:source,effect,strength:+$('strength').value/100,landmarks:$('landmarks').checked});$('result').src=j.image;$('faces').textContent=j.faces;$('latency').textContent=j.pipeline_ms+' ms';$('fps').textContent=j.pipeline_fps+' FPS';$('roundtrip').textContent=Math.round(performance.now()-t)+' ms';$('error').textContent=''}catch(e){$('error').textContent=e.message}finally{busy=false;if(pending&&!stream){pending=false;process()}}}
async function sample(){stop();try{source=await read(await(await fetch('/sample.jpg')).blob());$('source-label').textContent='公开示例 · NASA 宇航员照片';await process()}catch(e){$('error').textContent=e.message}}
function stop(){cameraGeneration++;if(stream){stream.getTracks().forEach(t=>t.stop());stream=null}$('video').srcObject=null;$('camera').hidden=false;$('camera').disabled=false;$('stop').hidden=true;pending=false}
async function live(){if(!stream)return;let v=$('video'),c=$('capture');if(v.videoWidth){c.width=640;c.height=Math.round(640*v.videoHeight/v.videoWidth);c.getContext('2d').drawImage(v,0,0,c.width,c.height);source=c.toDataURL('image/jpeg',.8);await process()}if(stream)requestAnimationFrame(live)}
$('effects').addEventListener('click',e=>{if(!e.target.dataset.effect)return;effect=e.target.dataset.effect;document.querySelectorAll('[data-effect]').forEach(b=>{b.classList.toggle('active',b===e.target);b.setAttribute('aria-pressed',b===e.target)});process()});
$('strength').oninput=()=>{$('amount').value=$('strength').value+'%';process()};$('landmarks').onchange=process;
$('upload').onchange=async()=>{stop();try{source=await read($('upload').files[0]);$('source-label').textContent='本机图像 · 仅内存处理';await process()}catch(e){$('error').textContent=e.message}};
$('camera').onclick=async()=>{const generation=++cameraGeneration;$('camera').disabled=true;try{const acquired=await navigator.mediaDevices.getUserMedia({video:{width:640,height:480},audio:false});if(generation!==cameraGeneration){acquired.getTracks().forEach(t=>t.stop());return}stream=acquired;$('video').srcObject=stream;await $('video').play();if(generation!==cameraGeneration)return;$('camera').hidden=true;$('stop').hidden=false;$('source-label').textContent='本机摄像头 · 实时画面';live()}catch(e){if(generation===cameraGeneration){stop();$('error').textContent='摄像头未开启：'+e.message}}};
$('stop').onclick=stop;$('sample').onclick=sample;$('save').onclick=()=>{let a=document.createElement('a');a.href=$('result').src;a.download='face-vision-result.jpg';a.click()};
$('verify').onclick=async()=>{try{$('verify').disabled=true;let j=await request('/verify',{first:await read($('first').files[0]),second:await read($('second').files[0]),threshold:+$('threshold').value});$('comparison').textContent=`余弦相似度 ${j.cosine.toFixed(4)}，阈值 ${j.threshold}：${j.match?'达到阈值':'未达到阈值'}。${j.note}`}catch(e){$('comparison').textContent=e.message}finally{$('verify').disabled=false}};
window.addEventListener('pagehide',stop);sample();
</script></html>

````
