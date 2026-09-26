"""Fetch original-owner CelebA/COFW files with published MD5 verification.

No authentication, quota bypass, mirror substitution, or identity form filling.
Official sources are linked in the output report. Data must remain local.
"""
import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
import zipfile
from html.parser import HTMLParser
from pathlib import Path

CELEBA=[
    ('img_align_celeba.zip','0B7EVK8r0v71pZjFTYXZWM3FlRnM','00d2c5bc6d35e252742224ab0c1e8fcb'),
    ('list_attr_celeba.txt','0B7EVK8r0v71pblRyaVFSWGxPY0U','75e246fa4810816ffd6ee81facbd244c'),
    ('list_eval_partition.txt','0B7EVK8r0v71pY0NSMzRuSXJEVkk','d32c9cbf5e040fd4025c592c306e6668')]
COFW=[('COFW.zip','e092eeab9d0790674f86047880410e5a'),('COFW_color.zip','8b21d126c4e1fb307cb463578eef0511'),('documentation.zip','cc3a8f6fc6b49f6186985c1c68c6fa52')]
STARGAN=[('celeba.zip','https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=1',None),
    ('celeba-128x128-5attrs.zip','https://www.dropbox.com/s/7e966qq0nlxwte4/celeba-128x128-5attrs.zip?dl=1',None)]


class DownloadForm(HTMLParser):
    def __init__(self): super().__init__(); self.action=None; self.values={}
    def handle_starttag(self,tag,attrs):
        values=dict(attrs)
        if tag=='form' and values.get('id')=='download-form': self.action=values.get('action')
        if tag=='input' and values.get('type')=='hidden' and 'name' in values: self.values[values['name']]=values.get('value','')


def response_for(url,headers=None):
    response=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'face-vision-lab/1.0',**(headers or {})}),timeout=90)
    if 'text/html' not in response.headers.get('Content-Type',''): return response
    page=response.read(1024*1024).decode('utf-8',errors='replace'); response.close()
    if 'Quota exceeded' in page or 'Too many users' in page:
        raise RuntimeError('Official Google Drive download quota exceeded; no bypass attempted')
    if 'Virus scan warning' in page:
        form=DownloadForm(); form.feed(page)
        if form.action!='https://drive.usercontent.google.com/download' or not form.values.get('id'):
            raise RuntimeError('Unexpected download form; refusing to invent parameters')
        # Standard public large-file confirmation, not a CAPTCHA or quota bypass.
        confirmed=form.action+'?'+urllib.parse.urlencode(form.values)
        response=urllib.request.urlopen(confirmed,timeout=90)
        if 'text/html' not in response.headers.get('Content-Type',''): return response
        message=response.read(10000).decode('utf-8',errors='replace'); response.close()
        raise RuntimeError('Official large-file confirmation did not yield a file: '+('quota exceeded' if 'Quota' in message or 'Too many' in message else 'HTML response'))
    raise RuntimeError('Official download returned HTML instead of a data file; authentication/registration may be required')


def download(url,destination,expected_md5):
    destination=Path(destination); destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.is_file():
        with destination.open('rb') as file: actual=hashlib.file_digest(file,'md5').hexdigest()
        if actual==expected_md5 or expected_md5 is None:
            with destination.open('rb') as file: sha256=hashlib.file_digest(file,'sha256').hexdigest()
            return {'status':'already_verified' if expected_md5 else 'already_present_no_published_archive_hash','bytes':destination.stat().st_size,'md5':actual,'sha256':sha256}
        raise RuntimeError(f'Existing {destination.name} has unexpected checksum; preserve it for inspection')
    partial=destination.with_name(destination.name+'.part')
    digest=hashlib.md5(usedforsecurity=False); sha=hashlib.sha256(); total=0; milestone=0; start=time.perf_counter()
    resume=partial.stat().st_size if partial.exists() and 'data.caltech.edu/' in url else 0
    with response_for(url,{'Range':f'bytes={resume}-'} if resume else None) as response:
        if resume and response.status==206:
            if not response.headers.get('Content-Range','').startswith(f'bytes {resume}-'): raise RuntimeError('Wrong resume byte range')
            with partial.open('rb') as old:
                while chunk:=old.read(1024*1024): digest.update(chunk); sha.update(chunk); total+=len(chunk)
        else: resume=0
        expected_length=int(response.headers['Content-Length'])+resume if response.headers.get('Content-Length') else None
        print(f'Start {destination.name}: HTTP {response.status}, expected bytes={response.headers.get("Content-Length")}, resume={resume}',flush=True)
        with partial.open('ab' if resume else 'wb') as file:
            reader=getattr(response,'read1',response.read)
            while chunk:=reader(1024*1024):
                file.write(chunk); digest.update(chunk); sha.update(chunk); total+=len(chunk)
                if total-milestone>=10*1024*1024:
                    print(f'{destination.name}: {total/1024/1024:.0f} MiB, {time.perf_counter()-start:.1f}s',flush=True); milestone=total
    if expected_length is not None and total!=expected_length:
        raise IOError(f'Truncated transfer for {destination.name}: {total} of {expected_length} bytes; partial preserved for ordinary Range resume')
    if expected_md5 is not None and digest.hexdigest()!=expected_md5: raise ValueError(f'Published MD5 mismatch for {destination.name}; partial preserved')
    partial.replace(destination)
    return {'status':'downloaded_verified' if expected_md5 else 'downloaded_hash_recorded_no_published_archive_hash',
        'bytes':total,'md5':digest.hexdigest(),'sha256':sha.hexdigest(),'seconds':time.perf_counter()-start}


def extract(archive,root):
    root=Path(root).resolve()
    with zipfile.ZipFile(archive) as file:
        for member in file.infolist():
            destination=(root/member.filename).resolve()
            if not destination.is_relative_to(root): raise ValueError('Unsafe archive member')
        for index,member in enumerate(file.infolist()):
            destination=(root/member.filename).resolve()
            # Resume our interrupted extraction; an interrupted current member has
            # a short size and is rewritten. Training manifests separately hash inputs.
            if not (destination.is_file() and destination.stat().st_size==member.file_size): file.extract(member,root)
            if (index+1)%10000==0: print(f'Extract {Path(archive).name}: {index+1}/{len(file.infolist())}',flush=True)


def run(args):
    root=Path(args.output or 'data/'+args.dataset)
    files=[(name,'https://drive.usercontent.google.com/download?'+urllib.parse.urlencode({'id':identifier,'export':'download','authuser':'0'}),md5)
        for name,identifier,md5 in CELEBA] if args.dataset=='celeba' else [
        (name,f'https://data.caltech.edu/records/bc0bf-nc666/files/{name}?download=1',md5) for name,md5 in COFW]
    if args.dataset=='stargan-official': files=STARGAN
    if args.file: files=[item for item in files if item[0]==args.file]
    if not files: raise ValueError('Unknown dataset filename')
    report={'dataset':args.dataset,'official_page':'https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html' if args.dataset=='celeba' else 'https://data.caltech.edu/records/bc0bf-nc666',
        'checksum_source':'https://github.com/pytorch/vision/blob/main/torchvision/datasets/celeba.py' if args.dataset=='celeba' else 'Official CaltechDATA record file list',
        'distribution':'Local research only; dataset is not included in the source-code repository','files':{}}
    if args.dataset=='stargan-official':
        report['official_page']='https://github.com/yunjey/stargan/blob/master/download.sh'
        report['checksum_source']='Algorithm authors do not publish bundle hashes; observed hashes and ZIP CRC are recorded, not original CelebA archive MD5 verification'
    for name,url,md5 in files:
        try:
            result=download(url,root/name,md5)
            if args.extract and name.endswith('.zip'): extract(root/name,root)
            report['files'][name]={'source_url':url,**result}; print(name,result,flush=True)
        except Exception as error:
            report['files'][name]={'source_url':url,'status':'unavailable','error':f'{type(error).__name__}: {error}'}
            print(name,report['files'][name]['error'],flush=True)
    destination=Path(args.report or f'reports/{args.dataset}-download.json'); destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('dataset',choices=['celeba','cofw','stargan-official'])
    parser.add_argument('--output'); parser.add_argument('--file'); parser.add_argument('--report'); parser.add_argument('--extract',action='store_true')
    result=run(parser.parse_args())
    if any(item['status']=='unavailable' for item in result['files'].values()): raise SystemExit(1)
