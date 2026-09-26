"""Download a reproducible real WIDER FACE pilot subset via HTTP ZIP ranges.

python -m research.fetch_wider_subset --train 32 --val 16 --seed 42
The dataset remains under its original CC BY-NC-ND-4.0 terms, not this code's
license. Data stay local. Small pilot COCO AP is NOT official WIDER benchmark AP.
"""
import argparse
import hashlib
import io
import json
import random
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from research.data import wider_to_coco

REPOSITORY='CUHK-CSE/wider_face'


def fetch(url,headers=None,limit=32*1024*1024):
    for attempt in range(3):
        try:
            request=urllib.request.Request(url,headers={'Accept-Encoding':'identity',**(headers or {})})
            with urllib.request.urlopen(request,timeout=60) as response:
                data=response.read(limit+1)
                if len(data)>limit: raise ValueError('Response exceeds bounded download size')
                return data,response.status,response.headers
        except (urllib.error.URLError,TimeoutError):
            if attempt==2: raise
            time.sleep(attempt+1)


class RemoteZipFile(io.RawIOBase):
    """Minimal seekable HTTP file for ZipFile; rejects servers ignoring Range."""
    def __init__(self,url,size):
        self.url,self.size,self.position=url,size,0
        self.cache_start=0; self.cache=b''; self.downloaded=0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.position

    def seek(self,offset,whence=0):
        position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if whence not in {0,1,2} or not 0<=position<=self.size: raise ValueError('Invalid remote seek')
        self.position=position; return position

    def read(self,size=-1):
        size=self.size-self.position if size<0 else min(size,self.size-self.position)
        if size==0: return b''
        if size>32*1024*1024: raise ValueError('ZIP member exceeds 32 MiB range-read guard')
        if not self.cache_start<=self.position or self.position+size>self.cache_start+len(self.cache):
            end=min(self.size,self.position+max(size,256*1024))-1
            expected=f'bytes {self.position}-{end}/{self.size}'
            data,status,headers=fetch(self.url,{'Range':f'bytes={self.position}-{end}'},end-self.position+1)
            if status!=206 or headers.get('Content-Range')!=expected or len(data)!=end-self.position+1:
                raise RuntimeError('Server ignored/mismatched HTTP Range; refusing a silent full-archive download')
            self.cache_start,self.cache=self.position,data; self.downloaded+=len(data)
        start=self.position-self.cache_start; self.position+=size
        return self.cache[start:start+size]


def annotation_blocks(text):
    lines=text.strip().splitlines(); cursor=0; result={}
    while cursor<len(lines):
        start=cursor; name=lines[cursor].strip(); cursor+=1
        if name in result or cursor>=len(lines): raise ValueError('Duplicate/truncated official annotations')
        count=int(lines[cursor]); cursor+=1
        if count<0 or cursor+count>len(lines): raise ValueError('Invalid official box count')
        cursor+=count
        if count==0 and cursor<len(lines):
            fields=lines[cursor].split()
            if len(fields)==10 and all(value=='0' for value in fields): cursor+=1
        result[name]='\n'.join(lines[start:cursor])+'\n'
    return result


def run(args):
    if min(args.train,args.val)<1: raise ValueError('Both train and val subset counts must be positive')
    root=Path(args.output).resolve(); root.mkdir(parents=True,exist_ok=True)
    metadata=json.loads(fetch(f'https://huggingface.co/api/datasets/{REPOSITORY}')[0])
    revision=metadata['sha']
    tree=json.loads(fetch(f'https://huggingface.co/api/datasets/{REPOSITORY}/tree/{revision}/data')[0])
    entries={Path(entry['path']).name:entry for entry in tree if entry['type']=='file'}
    base=f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{revision}/data'
    annotation_bytes,_,_=fetch(base+'/wider_face_split.zip')
    expected=entries['wider_face_split.zip']['lfs']['oid']
    if hashlib.sha256(annotation_bytes).hexdigest()!=expected: raise ValueError('Official annotation archive SHA256 mismatch')
    report={'dataset':REPOSITORY,'revision':revision,'license':'CC-BY-NC-ND-4.0',
        'purpose':'Real tiny training/validation pilot; not a standard benchmark or synthetic fixture',
        'selection':'Uniform sample without replacement from sorted annotated image names, independently within official train/val splits',
        'seed':args.seed,'annotations_sha256_verified':expected,'splits':{}}
    with zipfile.ZipFile(io.BytesIO(annotation_bytes)) as annotations:
        for split,count in [('train',args.train),('val',args.val)]:
            member=next(name for name in annotations.namelist() if name.endswith(f'wider_face_{split}_bbx_gt.txt'))
            blocks=annotation_blocks(annotations.read(member).decode('utf-8'))
            if count>len(blocks): raise ValueError('Requested more examples than split contains')
            selected=sorted(random.Random(args.seed).sample(sorted(blocks),count))
            archive_name=f'WIDER_{split}.zip'; entry=entries[archive_name]
            remote=RemoteZipFile(base+'/'+archive_name,entry['size'])
            image_root=root/f'WIDER_{split}'/'images'; image_root.mkdir(parents=True,exist_ok=True)
            hashes={}
            with zipfile.ZipFile(remote) as archive:
                for index,name in enumerate(selected):
                    destination=(image_root/name).resolve()
                    if not destination.is_relative_to(image_root): raise ValueError('Unsafe image archive path')
                    zip_name=f'WIDER_{split}/images/{name}'
                    if archive.getinfo(zip_name).file_size>32*1024*1024: raise ValueError('Uncompressed image exceeds size guard')
                    content=archive.read(zip_name)  # zipfile verifies each member's CRC32.
                    destination.parent.mkdir(parents=True,exist_ok=True); destination.write_bytes(content)
                    hashes[name]=hashlib.sha256(content).hexdigest()
                    print(f'WIDER {split}: {index+1}/{count}',flush=True)
            annotation_path=root/'wider'/f'subset_{split}_bbx_gt.txt'; annotation_path.parent.mkdir(parents=True,exist_ok=True)
            annotation_path.write_text(''.join(blocks[name] for name in selected),encoding='utf-8')
            conversion=wider_to_coco(image_root,annotation_path,root/'wider'/f'{split}.json')
            report['splits'][split]={**conversion,'image_sha256':hashes,'http_bytes_downloaded':remote.downloaded,
                'source_archive_sha256_advertised_not_fully_verified':entry['lfs']['oid'],
                'integrity':'Pinned repository revision; each downloaded ZIP member CRC32 verified; local images individually SHA256 recorded'}
    destination=Path(args.report); destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({name:{k:v for k,v in value.items() if k!='image_sha256'} for name,value in report['splits'].items()},indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',type=int,default=32); parser.add_argument('--val',type=int,default=16)
    parser.add_argument('--seed',type=int,default=42); parser.add_argument('--output',default='data')
    parser.add_argument('--report',default='reports/wider-subset.json')
    run(parser.parse_args())
