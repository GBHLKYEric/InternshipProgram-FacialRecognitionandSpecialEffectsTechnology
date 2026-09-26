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
