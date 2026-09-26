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
