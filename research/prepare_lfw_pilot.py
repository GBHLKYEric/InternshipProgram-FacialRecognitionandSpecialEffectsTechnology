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
