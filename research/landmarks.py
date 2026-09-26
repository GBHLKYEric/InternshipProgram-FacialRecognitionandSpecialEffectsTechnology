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
