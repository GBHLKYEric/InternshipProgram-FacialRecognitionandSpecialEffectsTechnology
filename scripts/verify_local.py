"""HTTP smoke and boundary check of the running local app, without a browser."""
import base64
import argparse
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


def main(port=8765, require_stargan=False):
    global BASE
    if not 1 <= port <= 65535: raise ValueError('Invalid TCP port')
    BASE=f'http://127.0.0.1:{port}'
    checks=[{'base_url':BASE}]
    for path in ['/','/health','/tutorial','/tutorial.html','/code-map.html',
                 '/code-compendium.html','/project-report.html','/3d','/sample.jpg']:
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
    available=json.loads(request('/health')[1]).get('stargan_available',False)
    if require_stargan: assert available,'Expected exported StarGAN model'
    if available:
        status,body,_=request('/attributes',{'image':image,'targets':[0,1,0,0,1]})
        edited=json.loads(body)
        assert status==200 and edited['image'].startswith('data:image/jpeg;base64,') and edited['model_ms']>0
        assert edited['image']!=edited['input_crop'],'GAN output unexpectedly equals input'
        checks.append({'path':'/attributes','status':status,'targets':edited['targets'],'model_ms':edited['model_ms']})
    else:
        checks.append({'path':'/attributes','status':'not tested: optional model absent'})
    assert request('/attributes',{'image':image,'targets':[1,1,0,0,1]})[0]==400
    for value in [{'image':'invalid'},{'image':image,'effect':'invalid'},{'image':image,'strength':3},[]]:
        assert request('/process',value)[0]==400
    assert request('/process',{'image':image},{'Origin':'https://untrusted.example'})[0]==403
    checks.append({'invalid_inputs':'400','cross_origin':'403','unknown_path':'404'})
    (ROOT/'reports/http-verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--require-stargan',action='store_true')
    args=parser.parse_args();main(args.port,args.require_stargan)
