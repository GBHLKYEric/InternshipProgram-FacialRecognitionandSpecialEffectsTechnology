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
