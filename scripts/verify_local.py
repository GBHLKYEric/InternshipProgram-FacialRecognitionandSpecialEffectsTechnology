"""HTTP smoke and boundary check of the running local app, without a browser."""
import base64
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


def main():
    checks=[]
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
    for value in [{'image':'invalid'},{'image':image,'effect':'invalid'},{'image':image,'strength':3},[]]:
        assert request('/process',value)[0]==400
    assert request('/process',{'image':image},{'Origin':'https://untrusted.example'})[0]==403
    checks.append({'invalid_inputs':'400','cross_origin':'403','unknown_path':'404'})
    (ROOT/'reports/http-verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))


if __name__=='__main__': main()
