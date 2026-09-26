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
