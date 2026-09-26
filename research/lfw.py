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
