"""Fetch complete official WIDER train/val archives with resumable ranges.

The author links this CUHK-CSE mirror. Every full image/annotation ZIP must
match its pinned LFS SHA256 before extraction. Original research data stay
local, under their own license. The small pilot JSON files are preserved.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import time
import urllib.request
import zipfile

from research.data import wider_to_coco
from research.fetch_wider_subset import fetch, annotation_blocks

REPOSITORY = 'CUHK-CSE/wider_face'
REVISION = 'db171f1b7fedf4d3453e81297ff02f9915356d19'
AUTHOR = 'http://shuoyang1213.me/WIDERFACE/'
EVAL_URL = 'http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/eval_script/eval_tools.zip'


def filesystem_path(path):
    """Use Windows extended paths locally, without changing system policy."""
    absolute = str(Path(path).resolve())
    if os.name == 'nt' and not absolute.startswith('\\\\?\\'):
        absolute = '\\\\?\\' + absolute
    return Path(absolute)


def sha256(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def download(url, destination, expected_size, expected_sha, chunk=8*1024*1024):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if destination.stat().st_size == expected_size and sha256(destination) == expected_sha:
            print(f'Verified existing {destination.name}', flush=True)
            return
        raise ValueError(f'Existing completed archive does not match: {destination}')
    partial = destination.with_suffix(destination.suffix + '.incomplete')
    position = partial.stat().st_size if partial.exists() else 0
    if position > expected_size:
        raise ValueError('Partial file exceeds source size; preserve it for inspection')
    start = time.perf_counter()
    initial = position
    with partial.open('ab') as output:
        while position < expected_size:
            end = min(expected_size, position + chunk) - 1
            data, status, headers = fetch(url, {'Range': f'bytes={position}-{end}'}, end-position+1)
            if status != 206 or headers.get('Content-Range') != f'bytes {position}-{end}/{expected_size}':
                raise ValueError('Server did not honor the exact requested byte range')
            if len(data) != end-position+1:
                raise ValueError('Truncated HTTP range')
            output.write(data)
            output.flush()
            position = end+1
            if position == expected_size or position//chunk % 8 == 0:
                speed = (position-initial)/max(time.perf_counter()-start, .001)/1e6
                print(f'{destination.name}: {position}/{expected_size} bytes, {speed:.2f} MB/s', flush=True)
    actual = sha256(partial)
    if actual != expected_sha:
        raise ValueError(f'SHA256 mismatch; incomplete file preserved: {actual}')
    partial.replace(destination)


def safe_extract(archive_path, root):
    root = filesystem_path(root)
    total = count = 0
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            destination = (root/member.filename).resolve()
            mode = member.external_attr >> 16
            if not destination.is_relative_to(root) or stat.S_ISLNK(mode):
                raise ValueError('Unsafe archive member path or symbolic link')
            if member.file_size > 128*1024*1024:
                raise ValueError('Archive member exceeds 128MiB safety limit')
            total += member.file_size
            if total > 10*1024**3:
                raise ValueError('Archive exceeds 10GiB expanded safety limit')
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            content = archive.read(member)  # checks CRC for every member
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and destination.read_bytes() != content:
                raise ValueError(f'Different existing file: {destination}')
            if not destination.exists():
                destination.write_bytes(content)
            count += 1
    return {'files': count, 'uncompressed_bytes': total, 'all_member_crc_verified': True}


def run(args):
    root = filesystem_path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {'source_author_page': AUTHOR, 'repository': REPOSITORY, 'revision': REVISION,
              'license': 'CC-BY-NC-ND-4.0; local research use; no redistribution of face images',
              'archives': {}, 'splits': {}}
    tree = json.loads(fetch(f'https://huggingface.co/api/datasets/{REPOSITORY}/tree/{REVISION}/data')[0])
    entries = {Path(entry['path']).name: entry for entry in tree if entry['type'] == 'file'}
    names = ['wider_face_split.zip'] + [f'WIDER_{split}.zip' for split in args.splits]
    # The author-published evaluation bundle supplies all three difficulty masks.
    eval_path = root/'wider-full/archives/eval_tools.zip'
    if not eval_path.exists():
        content, _, _ = fetch(EVAL_URL, limit=64*1024*1024)
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_path.write_bytes(content)
    report['evaluation_bundle'] = {'url': EVAL_URL, 'sha256_recorded': sha256(eval_path),
        'publisher_sha256_available': False, **safe_extract(eval_path, root)}
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name in names:
        entry = entries[name]
        url = f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/data/{name}'
        destination = root/'wider-full/archives'/name
        download(url, destination, entry['size'], entry['lfs']['oid'])
        report['archives'][name] = {'url': url, 'bytes': entry['size'],
            'sha256': entry['lfs']['oid'], 'publisher_sha256_verified': True,
            **safe_extract(destination, root)}
        if name.startswith('WIDER_'):
            split = name[6:-4]
            image_root = root/f'WIDER_{split}/images'
            expected = {'train': 12880, 'val': 3226}[split]
            count = sum(1 for _ in image_root.glob('*/*.jpg'))
            if count != expected:
                raise ValueError(f'Incomplete {split}: {count}, expected {expected}')
            report['splits'][split] = wider_to_coco(image_root,
                root/f'wider_face_split/wider_face_{split}_bbx_gt.txt', root/f'wider/full_{split}.json')
            blocks = annotation_blocks((root/f'wider_face_split/wider_face_{split}_bbx_gt.txt').read_text())
            report['splits'][split]['zero_face_images'] = sum(int(block.splitlines()[1]) == 0 for block in blocks.values())
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'Extracted and recorded {name}', flush=True)
    return report


if __name__ == '__main__':
    if os.name == 'nt':
        os.environ.setdefault('SystemRoot', r'C:\Windows')
        os.environ.setdefault('WINDIR', r'C:\Windows')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='data')
    parser.add_argument('--splits', nargs='+', choices=['val', 'train'], default=['val', 'train'])
    parser.add_argument('--report', default='reports/wider-full-data.json')
    run(parser.parse_args())
