"""Export a trained StarGAN generator and compare ONNX with real-image PyTorch."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
import torch
from torchvision import transforms
from research.stargan import load_generator


def export(args):
    torch.set_num_threads(args.threads)
    model,config=load_generator(args.checkpoint)
    attributes=['Black_Hair','Blond_Hair','Brown_Hair','Male','Young']
    if config['attributes']!=attributes or config['size']!=128:
        raise ValueError('Deployment requires the documented five-attribute 128-pixel model')
    targets=[float(value) for value in args.targets.split(',')]
    if len(targets)!=5 or not set(targets)<={0.,1.} or sum(targets[:3])>1:
        raise ValueError('Five binary targets are required, at most one of three hair colors')
    with Image.open(args.image) as source:
        image=transforms.Compose([transforms.CenterCrop(178),transforms.Resize((128,128)),
            transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])(source.convert('RGB'))[None]
    conditions=torch.tensor([targets],dtype=torch.float32)
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        expected=model(image,conditions).numpy()
        torch.onnx.export(model,(image,conditions),str(output),input_names=['images','attributes'],
            output_names=['edited'],opset_version=17,dynamo=False)
    onnx.checker.check_model(onnx.load(output))
    options=ort.SessionOptions();options.intra_op_num_threads=args.threads;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(output),sess_options=options,providers=['CPUExecutionProvider'])
    feed={'images':image.numpy(),'attributes':conditions.numpy()}
    actual=session.run(['edited'],feed)[0]
    np.testing.assert_allclose(actual,expected,rtol=1e-3,atol=1e-4)
    milliseconds=[]
    for _ in range(args.repeats):
        start=time.perf_counter();session.run(['edited'],feed);milliseconds.append((time.perf_counter()-start)*1000)
    with Path(args.checkpoint).open('rb') as file: checkpoint_sha=hashlib.file_digest(file,'sha256').hexdigest()
    with output.open('rb') as file: model_sha=hashlib.file_digest(file,'sha256').hexdigest()
    report={'model':str(output),'checkpoint':str(args.checkpoint),'checkpoint_sha256':checkpoint_sha,
        'model_sha256':model_sha,'model_bytes':output.stat().st_size,'input_names':['images','attributes'],
        'input_shapes':[[1,3,128,128],[1,5]],'output_name':'edited','output_shape':list(actual.shape),
        'dtype':'float32','model_mode':'eval; InstanceNorm uses per-image statistics, matching the author Solver.test behavior; obsolete running buffers are disabled after strict weight loading',
        'attributes':attributes,'comparison_real_image':str(args.image),'comparison_targets':targets,
        'max_abs_error':float(np.max(np.abs(actual-expected))),'mean_abs_error':float(np.mean(np.abs(actual-expected))),
        'allclose_tolerance':{'rtol':.001,'atol':.0001},'ort_median_ms':float(np.median(milliseconds)),
        'ort_repeats':args.repeats,'cpu_threads':args.threads,'torch':torch.__version__,'onnx':onnx.__version__,
        'latency_caveat':'Measured during project work on this machine, not an isolated hardware benchmark.',
        'onnxruntime':ort.__version__,'provider':'CPUExecutionProvider',
        'preprocessing':'RGB; center crop 178x178 (PIL zero-padding if smaller); bilinear resize128x128; NCHW float32; pixel/127.5-1',
        'postprocessing':'Clip output to [-1,1]; convert (output+1)*127.5 to RGB uint8.',
        'scope':'Numerical deployment equivalence on one real held-out image; not an attribute-quality or identity-preservation certificate.'}
    destination=Path(args.report);destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--image',required=True)
    parser.add_argument('--targets',default='0,1,0,0,1');parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--repeats',type=int,default=5);parser.add_argument('--output',default='runs/stargan-deploy/generator.onnx')
    parser.add_argument('--report',default='reports/stargan-deployment.json');export(parser.parse_args())
