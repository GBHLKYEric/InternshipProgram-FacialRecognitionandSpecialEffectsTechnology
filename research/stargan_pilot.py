"""Reproduce the StarGAN authors' split, then prepare/train/evaluate a local pilot.

Data originate from the StarGAN official project's supporting download, not a
CelebA-owner mirror. All images and generated grids remain under runs/ or data/.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import zipfile
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
from research.stargan import load_generator,metrics


def prepare(args):
    with zipfile.ZipFile(args.archive) as archive: raw=archive.read('celeba/list_attr_celeba.txt')
    md5=hashlib.md5(raw).hexdigest()
    if md5!='75e246fa4810816ffd6ee81facbd244c': raise ValueError('Attributes do not match torchvision-published original CelebA MD5')
    lines=raw.decode('utf-8').splitlines(); rows=[line.rstrip() for line in lines[2:]]
    if int(lines[0])!=202599 or len(rows)!=202599: raise ValueError('Expected original CelebA 202599 annotations')
    attr_names=lines[1].split()
    positives=[0]*len(attr_names)
    for row in rows:
        for column,value in enumerate(row.split()[1:]): positives[column]+=value=='1'
    positive_counts=dict(zip(attr_names,positives))
    with zipfile.ZipFile(args.archive) as archive:
        image_names={Path(name).name for name in archive.namelist() if name.startswith('celeba/images/') and name.endswith('.jpg')}
    if image_names!={row.split()[0] for row in rows}: raise ValueError('Complete archive image names do not match annotations')
    random.Random(1234).shuffle(rows)  # Exactly mirrors official StarGAN data_loader.py.
    if args.train<2 or args.evaluation<10 or args.evaluation*2>1999: raise ValueError('Use train>=2 and 10<=evaluation<=999')
    groups={'train':rows[1999:1999+args.train],'generation_source':rows[:args.evaluation],
        'real_reference':rows[args.evaluation:2*args.evaluation]}
    root=Path(args.images).resolve(); output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    entries={}; selections=[]; partition=[]
    with zipfile.ZipFile(args.archive) as source_archive:
        for split_index,(split,chosen) in enumerate(groups.items()):
            entries[split]=[]
            for row in chosen:
                name=row.split()[0]; path=(root/name).resolve()
                if not path.is_relative_to(root): raise ValueError(f'Unsafe image: {name}')
                image_bytes=source_archive.read('celeba/images/'+name)  # Verifies member CRC32.
                expected=hashlib.sha256(image_bytes).hexdigest()
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(image_bytes)
                entries[split].append({'name':name,'sha256':expected})
                selections.append(row); partition.append(f'{name} {split_index}')
    names=[item['name'] for items in entries.values() for item in items]
    if len(names)!=len(set(names)): raise ValueError('Training/generation/reference images overlap')
    (output/'attributes.txt').write_text(str(len(selections))+'\n'+lines[1]+'\n'+'\n'.join(selections)+'\n',encoding='utf-8')
    (output/'partition.txt').write_text('\n'.join(partition)+'\n',encoding='utf-8')
    manifest={'source':'StarGAN official project supporting download; https://github.com/yunjey/stargan/blob/master/download.sh',
        'original_celeba_attribute_md5_verified':md5,'original_image_archive_md5_verified':False,
        'full_corpus_image_count':len(image_names),'full_corpus_attribute_count':len(attr_names),
        'full_corpus_positive_attribute_counts':positive_counts,
        'split':'Reconstruct StarGAN authors data_loader.py: shuffle original annotation order with seed1234, first1999 held out; remainder training. NOT CelebA official partition file.',
        'pretrained_overlap_caveat':'Checkpoint actual historical training data were not independently audited; split reconstruction from author code does not prove its training provenance.',
        'selection':'First requested counts within reconstructed author splits; generation sources and real references are disjoint held-out images.',
        'images':str(root),'groups':entries,'training_images':args.train,'generation_images':args.evaluation,'real_reference_images':args.evaluation,
        'data_policy':'Local non-commercial research only. Do not publish these dataset images or original-image grids.'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    report=Path(args.report); report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print({k:len(v) for k,v in entries.items()})


def evaluate(args):
    torch.set_num_threads(args.threads)
    manifest=json.loads(Path(args.manifest).read_text(encoding='utf-8')); root=Path(manifest['images'])
    generator,checkpoint=load_generator(args.checkpoint)
    generator.to(args.device)
    attributes=checkpoint['attributes']
    lines=Path(args.attributes).read_text(encoding='utf-8').splitlines()
    names=lines[1].split(); columns=[names.index(name)+1 for name in attributes]
    labels={fields[0]:[float(fields[column]=='1') for column in columns] for line in lines[2:] if (fields:=line.split())}
    output=Path(args.output); generated=output/'generated'; real=output/'real-reference'
    generated.mkdir(parents=True,exist_ok=True); real.mkdir(parents=True,exist_ok=True)
    size=checkpoint['size']
    transform=transforms.Compose([transforms.CenterCrop(178),transforms.Resize((size,size)),
        transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])
    source_names=[item['name'] for item in manifest['groups']['generation_source']]
    reference_names=[item['name'] for item in manifest['groups']['real_reference']]
    train_names={item['name'] for item in manifest['groups']['train']}
    if len(source_names)!=len(reference_names) or len(source_names)!=len(set(source_names)) or len(reference_names)!=len(set(reference_names)):
        raise ValueError('Evaluation groups must have the same count and unique images')
    if set(source_names)&set(reference_names) or train_names&(set(source_names)|set(reference_names)):
        raise ValueError('Training/FID source/reference sets must be disjoint')
    for group in ['generation_source','real_reference']:
        for item in manifest['groups'][group]:
            path=(root/item['name']).resolve()
            if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
                raise ValueError('Evaluation image does not match the recorded manifest')
    for directory,expected_names in [(generated,source_names),(real,reference_names)]:
        expected={name.replace('.jpg','.png') for name in expected_names}
        if any(path.name not in expected for path in directory.iterdir()):
            raise ValueError('Evaluation output contains unrelated files; select a fresh directory')
    with torch.inference_mode():
        for start in range(0,len(source_names),4):
            sources=source_names[start:start+4]; references=reference_names[start:start+4]
            source_tensors=[]; reference_tensors=[]
            for filename in sources:
                with Image.open(root/filename) as image: source_tensors.append(transform(image.convert('RGB')))
            for filename in references:
                with Image.open(root/filename) as image: reference_tensors.append(transform(image.convert('RGB')))
            batch=torch.stack(source_tensors); target=torch.tensor([labels[name] for name in references])
            result=generator(batch.to(args.device),target.to(args.device)).cpu()
            for index,filename in enumerate(sources): save_image(result[index],generated/filename.replace('.jpg','.png'),normalize=True,value_range=(-1,1))
            for index,filename in enumerate(references): save_image(reference_tensors[index],real/filename.replace('.jpg','.png'),normalize=True,value_range=(-1,1))
            if start==0: save_image(torch.cat([batch,result]),output/'private-source-edited-grid.png',nrow=len(batch),normalize=True,value_range=(-1,1))
            if start%64==0 or start+4>=len(source_names): print(f'Generate held-out: {min(start+4,len(source_names))}/{len(source_names)}',flush=True)
    args.real=str(real); args.generated=str(generated); args.cuda=False
    args.output=str(output/'metrics.json'); metrics(args)
    result=json.loads(Path(args.output).read_text(encoding='utf-8'))
    result.update({'checkpoint':str(args.checkpoint),'generation_policy':'Target attribute vector matches corresponding disjoint real-reference image; source identities unchanged by intention, not verified.',
        'manifest':str(args.manifest),'source_reference_overlap':0,'generation_device':args.device,'metric_device':'cpu',
        'sample_size_warning':'Pilot FID/IS with limited samples; not comparable to paper-standard large-sample results.',
        'identity_overlap_caveat':'Disjoint image filenames do not imply disjoint people; official StarGAN splitting is by image, not identity.',
        'pretrained_overlap_caveat':manifest['pretrained_overlap_caveat']})
    with Path(args.checkpoint).open('rb') as file: result['checkpoint_sha256']=hashlib.file_digest(file,'sha256').hexdigest()
    Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    if args.report:
        report=Path(args.report);report.parent.mkdir(parents=True,exist_ok=True);report.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


def summarize(args):
    """Collect this documented, interrupted experiment without hiding discarded work."""
    os.environ.setdefault('MPLCONFIGDIR',str(Path('runs/matplotlib-cache').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    stages=[('pilot',Path('runs/stargan-pilot'),0,200),
        ('expanded_interrupted',Path('runs/stargan-expanded'),200,500),
        ('resumed_seed43',Path('runs/stargan-resumed'),700,500)]
    rows=[];configs={}
    for stage,directory,offset,kept_steps in stages:
        config=json.loads((directory/'config.json').read_text(encoding='utf-8'));configs[stage]=config
        shutil.copyfile(directory/'config.json',output/(stage+'-config.json'))
        with (directory/'history.csv').open(encoding='utf-8',newline='') as file:
            for row in csv.DictReader(file):
                row.update(stage=stage,retained=int(row['step'])<=kept_steps,
                    cumulative_retained_axis=offset+int(row['step']))
                rows.append(row)
    with (output/'training-history.csv').open('w',encoding='utf-8',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
    for stage,_,_,_ in stages:
        selected=[row for row in rows if row['stage']==stage and row['retained']]
        for column,label in [('d_loss','D'),('g_loss','G')]:
            axes[0].plot([row['cumulative_retained_axis'] for row in selected],
                [float(row[column]) for row in selected],label=stage+' '+label)
        axes[1].plot([row['cumulative_retained_axis'] for row in selected],
            [float(row['reconstruction_l1']) for row in selected],label=stage)
    for axis in axes:
        axis.axvline(200,color='grey',linestyle=':',alpha=.7)
        axis.axvline(700,color='grey',linestyle=':',alpha=.7)
        axis.grid(alpha=.2);axis.legend(fontsize=7)
    axes[0].set_ylabel('Actual logged loss (different protocol after 200)')
    axes[1].set_ylabel('Reconstruction mean absolute error')
    axes[1].set_xlabel('D updates retained in the final checkpoint')
    fig.suptitle('Real CelebA fine-tuning from author pretrained weights; discarded updates omitted from curves')
    fig.tight_layout();fig.savefig(output/'training-curves.svg');plt.close(fig)
    before=json.loads(Path('reports/stargan-before.json').read_text(encoding='utf-8'))
    after=json.loads(Path('reports/stargan-after.json').read_text(encoding='utf-8'))
    training=json.loads(Path('runs/stargan-resumed/training-report.json').read_text(encoding='utf-8'))
    initial=json.loads(Path('runs/stargan-pilot/training-report.json').read_text(encoding='utf-8'))
    deployment=json.loads(Path('reports/stargan-deployment.json').read_text(encoding='utf-8'))
    if training['cumulative_discriminator_steps']!=1200 or initial['steps']!=200 or training['steps']!=500:
        raise ValueError('This summarizer describes the documented 200+500+500 experiment; stage records differ')
    if after['checkpoint_sha256']!=deployment['checkpoint_sha256'] or before['manifest']!=after['manifest']:
        raise ValueError('Deployment/checkpoint or before/after evaluation protocols differ')
    for field in ['generation_device','metric_device','metric_torch_version','metric_library_version','metric_seed','metric_threads']:
        if before[field]!=after[field]: raise ValueError(f'Controlled comparison differs in {field}')
    report={'final_checkpoint':'runs/stargan-resumed/last.pt','retained_discriminator_updates':1200,
        'retained_generator_updates':240,'stage_retained_D_updates':[200,500,500],
        'stage_sample_draws':[400,1000,1000],'configured_training_pools':[512,4096,4096],
        'unique_training_images_seen_total':None,
        'sample_count_caveat':'Pool sizes and sample draws are not distinct people or total unique images; the two expanded-stage permutations can overlap.',
        'discarded_logged_stage_steps':[int(row['step']) for row in rows if not row['retained']],
        'discarded_updates_lower_bound':150,'unlogged_interruption_tail':'unknown; not counted in the retained model',
        'restoration':'Both models and both Adam states restored. Seed42 first segments; seed43 last segment starts a new shuffled data order.',
        'loss_protocol_change_at_retained_step':200,'first_stage':initial,'last_stage':training,
        'before':before,'after':after,
        'fid_after_minus_before':after['frechet_inception_distance']-before['frechet_inception_distance'],
        'is_mean_after_minus_before':after['inception_score_mean']-before['inception_score_mean'],
        'quality_claim':'Same 768 sources and 768 disjoint references; limited-sample observed comparison only. No statistical significance, identity-preservation guarantee, or paper-level reproduction claim.',
        'normalization':'Both before and after use per-image InstanceNorm; obsolete running statistics are disabled. Invalid running-statistics baseline was excluded.',
        'deployment_report':'reports/stargan-deployment.json','final_checkpoint_sha256':after['checkpoint_sha256']}
    historical=Path('reports/stargan-before-cpu.json')
    if historical.exists():
        report['historical_before_cpu']=json.loads(historical.read_text(encoding='utf-8'))
        report['historical_note']='First before run used CPU generation and CPU torch build; retained as history, excluded from the final matched-device/build comparison.'
    with Path('runs/torch-hub/checkpoints/weights-inception-2015-12-05-6726825d.pth').open('rb') as file:
        report['metric_inception_weights_sha256']=hashlib.file_digest(file,'sha256').hexdigest()
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({key:report[key] for key in ['retained_discriminator_updates','retained_generator_updates','fid_after_minus_before','is_mean_after_minus_before']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('prepare');p.add_argument('--archive',default='data/stargan-official/celeba.zip')
    p.add_argument('--images',default='data/stargan-official/celeba/images');p.add_argument('--output',default='data/celeba-pilot')
    p.add_argument('--train',type=int,default=512);p.add_argument('--evaluation',type=int,default=128)
    p.add_argument('--report',default='reports/celeba-pilot-data.json')
    p=commands.add_parser('evaluate');p.add_argument('--checkpoint',required=True)
    p.add_argument('--manifest',default='data/celeba-pilot/manifest.json');p.add_argument('--attributes',default='data/celeba-pilot/attributes.txt')
    p.add_argument('--output',default='runs/stargan-evaluation');p.add_argument('--report');p.add_argument('--threads',type=int,default=2);p.add_argument('--device',default='cpu')
    p=commands.add_parser('summarize');p.add_argument('--output',default='reports/stargan-experiment')
    args=parser.parse_args(); {'prepare':prepare,'evaluate':evaluate,'summarize':summarize}[args.command](args)
