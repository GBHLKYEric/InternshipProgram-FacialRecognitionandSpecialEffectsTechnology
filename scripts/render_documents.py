"""Render delivered PDFs to reviewable PNG pages and numbered contact sheets.

Convert the PPTX with an installed office renderer separately, then pass its PDF
through --slides-pdf. This tool checks appearance, not scientific correctness.
"""
import argparse
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/document-preview')
    parser.add_argument('--slides-pdf', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = [ROOT.parent/name for name in
               ['高三零基础专业教程.pdf', '项目总结报告.pdf', '执行问题与解决记录.pdf']]
    if args.slides_pdf:
        sources.append(args.slides_pdf)
    for source in sources:
        with pdfium.PdfDocument(str(source)) as document:
            previews = []
            for index in range(len(document)):
                page = document[index]
                bitmap = page.render(scale=1.15)
                picture = bitmap.to_pil().convert('RGB')
                picture.save(args.output/f'{source.stem}-{index+1:02d}.png')
                bitmap.close()
                page.close()
                picture.thumbnail((350, 480))
                tile = Image.new('RGB', (370, 510), '#dfe4e0')
                tile.paste(picture, ((370-picture.width)//2, 16))
                ImageDraw.Draw(tile).text((12, 490), str(index+1), fill='black')
                previews.append(tile)
            columns = 4
            rows = (len(previews)+columns-1)//columns
            montage = Image.new('RGB', (370*columns, 510*rows), '#cbd3cd')
            for index, picture in enumerate(previews):
                montage.paste(picture, ((index%columns)*370, (index//columns)*510))
            montage.save(args.output/f'{source.stem}-montage.jpg')
            print(source.name, len(previews))


if __name__ == '__main__':
    main()
