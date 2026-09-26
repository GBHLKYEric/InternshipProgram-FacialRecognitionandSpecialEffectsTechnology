"""Small, actual StarGAN-style WGAN-GP training/translation/FID+IS entry points.

This educational implementation is not pretrained and does not imply a successful
attribute editor until real CelebA training and held-out inspection are complete.
"""
import argparse
import csv
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from research.recognition import seed_all


class Residual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(width,width,3,1,1,bias=False), nn.InstanceNorm2d(width,affine=True),
            nn.ReLU(), nn.Conv2d(width,width,3,1,1,bias=False), nn.InstanceNorm2d(width,affine=True))

    def forward(self,x): return x+self.block(x)


class Generator(nn.Module):
    def __init__(self, attributes, width=64, blocks=6):
        super().__init__()
        layers=[nn.Conv2d(3+attributes,width,7,1,3,bias=False),nn.InstanceNorm2d(width,affine=True),nn.ReLU()]
        for _ in range(2):
            layers.extend([nn.Conv2d(width,width*2,4,2,1,bias=False),nn.InstanceNorm2d(width*2,affine=True),nn.ReLU()]); width*=2
        layers.extend([Residual(width) for _ in range(blocks)])
        for _ in range(2):
            layers.extend([nn.ConvTranspose2d(width,width//2,4,2,1,bias=False),nn.InstanceNorm2d(width//2,affine=True),nn.ReLU()]); width//=2
        self.network=nn.Sequential(*layers,nn.Conv2d(width,3,7,1,3),nn.Tanh())

    def forward(self,image,attributes):
        conditions=attributes[:,:,None,None].expand(-1,-1,image.shape[2],image.shape[3])
        return self.network(torch.cat([image,conditions],dim=1))


class Discriminator(nn.Module):
    def __init__(self,attributes,width=64,depth=5):
        super().__init__()
        layers=[]; channels=3
        for _ in range(depth):
            layers.extend([nn.Conv2d(channels,width,4,2,1),nn.LeakyReLU(.01)]); channels=width; width*=2
        self.features=nn.Sequential(*layers)
        self.source=nn.Conv2d(channels,1,3,1,1)
        self.classifier=nn.Linear(channels,attributes)

    def forward(self,image):
        features=self.features(image)
        return self.source(features).mean((1,2,3)),self.classifier(features.mean((2,3)))


class CelebA(Dataset):
    def __init__(self,root,attributes_file,partition_file,attributes,size):
        self.root=Path(root).resolve()
        lines=Path(attributes_file).read_text(encoding='utf-8').strip().splitlines()
        total=int(lines[0]); names=lines[1].split()
        if len(lines)-2!=total: raise ValueError('CelebA attribute count does not match header')
        if len(set(attributes))!=len(attributes) or not set(attributes)<=set(names): raise ValueError('Unknown/duplicate CelebA attributes')
        columns=[names.index(name) for name in attributes]
        partitions={}
        for line in Path(partition_file).read_text(encoding='utf-8').splitlines():
            name,split=line.split()
            if name in partitions or split not in {'0','1','2'}: raise ValueError('Invalid/duplicate partition entry')
            partitions[name]=split
        self.entries=[]; seen=set()
        for line in lines[2:]:
            fields=line.split(); name=fields[0]
            if name in seen or len(fields)!=len(names)+1 or any(value not in {'-1','1'} for value in fields[1:]):
                raise ValueError('Malformed CelebA attribute row')
            seen.add(name)
            if name not in partitions: raise ValueError('Every CelebA image needs a partition entry')
            if partitions[name]!='0': continue
            path=(self.root/name).resolve()
            if not path.is_relative_to(self.root) or not path.is_file(): raise FileNotFoundError(f'Missing training image: {path}')
            self.entries.append((path,torch.tensor([float(fields[column+1]=='1') for column in columns])))
        if len(self.entries)<2: raise ValueError('Need >=2 real CelebA training images')
        self.transform=transforms.Compose([transforms.CenterCrop(178),transforms.Resize((size,size)),
            transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])

    def __len__(self): return len(self.entries)

    def __getitem__(self,index):
        path,labels=self.entries[index]
        with Image.open(path) as image: tensor=self.transform(image.convert('RGB'))
        return tensor,labels


def gradient_penalty(discriminator,real,fake):
    alpha=torch.rand(real.shape[0],1,1,1,device=real.device)
    mixed=(alpha*real+(1-alpha)*fake).requires_grad_(True)
    scores,_=discriminator(mixed)
    gradient=torch.autograd.grad(scores.sum(),mixed,create_graph=True)[0]
    return (gradient.flatten(1).norm(2,dim=1)-1).square().mean()


def train(args):
    seed_all(args.seed)
    if args.size not in {64,128,256} or args.n_critic<1 or args.steps<args.n_critic or args.batch_size<2 or min(args.width,args.blocks,args.lr,args.log_interval,args.save_interval)<=0:
        raise ValueError('Use size 64/128/256, steps>=n_critic>=1, batch_size>=2, positive model/optimizer/logging settings')
    dataset=CelebA(args.root,args.labels,args.partition,args.attributes,args.size)
    loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True,drop_last=True)
    if not len(loader): raise ValueError('Batch size exceeds training data size')
    generator=Generator(len(args.attributes),args.width,args.blocks).to(args.device)
    discriminator=Discriminator(len(args.attributes),args.width).to(args.device)
    g_optimizer=torch.optim.Adam(generator.parameters(),lr=args.lr,betas=(.5,.999))
    d_optimizer=torch.optim.Adam(discriminator.parameters(),lr=args.lr,betas=(.5,.999))
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    iterator=iter(loader)
    with (output/'history.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.writer(file); writer.writerow(['step','d_loss','g_loss','reconstruction_l1'])
        for step in range(1,args.steps+1):
            try: real,original=next(iterator)
            except StopIteration: iterator=iter(loader); real,original=next(iterator)
            real,original=real.to(args.device),original.to(args.device)
            target=original[torch.randperm(len(original),device=args.device)]
            d_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(): fake=generator(real,target)
            real_score,real_class=discriminator(real); fake_score,_=discriminator(fake)
            d_loss=fake_score.mean()-real_score.mean()+F.binary_cross_entropy_with_logits(real_class,original)+10*gradient_penalty(discriminator,real,fake)
            if not torch.isfinite(d_loss): raise FloatingPointError('Non-finite discriminator loss')
            d_loss.backward(); d_optimizer.step()
            g_loss=None; reconstruction=None
            if step%args.n_critic==0:
                for parameter in discriminator.parameters(): parameter.requires_grad_(False)
                g_optimizer.zero_grad(set_to_none=True)
                fake=generator(real,target); score,classification=discriminator(fake)
                reconstruction=F.l1_loss(generator(fake,original),real)
                g_loss=-score.mean()+F.binary_cross_entropy_with_logits(classification,target)+10*reconstruction
                if not torch.isfinite(g_loss): raise FloatingPointError('Non-finite generator loss')
                g_loss.backward(); g_optimizer.step()
                for parameter in discriminator.parameters(): parameter.requires_grad_(True)
            if step%args.log_interval==0 or step==args.steps:
                row=[step,d_loss.item(),None if g_loss is None else g_loss.item(),None if reconstruction is None else reconstruction.item()]
                writer.writerow(row); file.flush(); print(row,flush=True)
            if step%args.save_interval==0 or step==args.steps:
                torch.save({'model_type':'educational_stargan','generator':generator.state_dict(),
                    'discriminator':discriminator.state_dict(),'g_optimizer':g_optimizer.state_dict(),
                    'd_optimizer':d_optimizer.state_dict(),'attributes':args.attributes,'width':args.width,
                    'blocks':args.blocks,'size':args.size,'step':step,'seed':args.seed},output/'last.pt')
                with torch.no_grad():
                    save_image(torch.cat([real[:4],generator(real[:4],target[:4])]),output/'latest_grid.png',nrow=min(4,len(real)),normalize=True,value_range=(-1,1))


def generate(args):
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if checkpoint.get('model_type')!='educational_stargan': raise ValueError('Expected this implementation checkpoint')
    targets=[int(value) for value in args.targets.split(',')]
    if len(targets)!=len(checkpoint['attributes']) or not set(targets)<={0,1}: raise ValueError('targets must have one 0/1 per checkpoint attribute')
    hair=[targets[index] for index,name in enumerate(checkpoint['attributes']) if name in {'Black_Hair','Blond_Hair','Brown_Hair','Gray_Hair'}]
    if sum(hair)>1: raise ValueError('Choose at most one mutually exclusive hair color')
    generator=Generator(len(targets),checkpoint['width'],checkpoint['blocks']).eval()
    generator.load_state_dict(checkpoint['generator'])
    with Image.open(args.image) as image:
        transform=transforms.Compose([transforms.CenterCrop(178),transforms.Resize((checkpoint['size'],)*2),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])
        tensor=transform(image.convert('RGB'))[None]
    destination=Path(args.output); destination.parent.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode(): result=generator(tensor,torch.tensor([targets],dtype=torch.float32))
    save_image(result,destination,normalize=True,value_range=(-1,1))


def metrics(args):
    import torch_fidelity
    counts=[]
    for directory in [args.real,args.generated]:
        files=[p for p in Path(directory).rglob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg'}]
        if len(files)<10: raise ValueError('Need at least ten images per set for ten IS splits; use thousands of held-out samples for meaningful estimates')
        counts.append(len(files))
    result=torch_fidelity.calculate_metrics(input1=args.generated,input2=args.real,cuda=args.cuda,
        isc=True,fid=True,kid=False,verbose=True)
    result['note']='Report image count/preprocessing/sampling policy; small-sample FID and IS are unreliable. IS is ImageNet-based and limited for faces.'
    result.update({'real_images':counts[0],'generated_images':counts[1],'real_directory':str(args.real),'generated_directory':str(args.generated)})
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('train')
    for name in ['root','labels','partition']: p.add_argument('--'+name,required=True)
    p.add_argument('--attributes',nargs='+',default=['Black_Hair','Blond_Hair','Brown_Hair','Male','Young'])
    p.add_argument('--output',default='runs/stargan'); p.add_argument('--device',default='cpu')
    for name,value in [('steps',100000),('batch-size',8),('size',128),('width',64),('blocks',6),('n-critic',5),('seed',42),('log-interval',10),('save-interval',1000)]: p.add_argument('--'+name,type=int,default=value)
    p.add_argument('--lr',type=float,default=.0001)
    p=commands.add_parser('generate')
    for name in ['checkpoint','image','targets','output']: p.add_argument('--'+name,required=True)
    p=commands.add_parser('metrics')
    for name in ['real','generated']: p.add_argument('--'+name,required=True)
    p.add_argument('--output',default='runs/stargan-metrics.json'); p.add_argument('--cuda',action='store_true')
    args=parser.parse_args(); {'train':train,'generate':generate,'metrics':metrics}[args.command](args)
