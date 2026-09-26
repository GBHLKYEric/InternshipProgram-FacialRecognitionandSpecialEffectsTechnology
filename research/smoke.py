"""Random-tensor gradient checks only. These are NOT training or benchmark results."""
import argparse
import json
import tempfile
from types import SimpleNamespace
from pathlib import Path
import torch
from research.recognition import ArcFace,EmbeddingNet,PKSampler,batch_hard_triplet,seed_all
from research.landmarks import make_model
from research.stargan import Generator,Discriminator,gradient_penalty


def end_to_end():
    """Exercise loaders/optimizers/checkpoints on explicit throwaway noise fixtures."""
    import numpy as np
    from PIL import Image
    from research.recognition import train as train_recognition,load_embedding
    from research.landmarks import train as train_landmarks,infer as infer_landmarks
    from research.stargan import train as train_gan,generate
    with tempfile.TemporaryDirectory(prefix='face-vision-synthetic-smoke-') as directory:
        root=Path(directory); rng=np.random.default_rng(42)
        for identity in ['synthetic_A','synthetic_B']:
            (root/'identities'/identity).mkdir(parents=True)
            for number in range(2):
                Image.fromarray(rng.integers(0,256,(178,178,3),dtype=np.uint8)).save(root/'identities'/identity/f'{number}.jpg')
        train_recognition(SimpleNamespace(data=str(root/'identities'),output=str(root/'arcface'),
            epochs=1,p=2,k=2,seed=42,workers=0,embedding_dim=32,device='cpu',lr=.001,triplet_weight=.1))
        model,_=load_embedding(root/'arcface/last.pt')
        with torch.inference_mode(): assert model(torch.randn(1,3,112,112)).shape==(1,32)
        entries=[]
        for number,path in enumerate(sorted((root/'identities').rglob('*.jpg'))):
            points=np.column_stack([np.linspace(30,140,68),80+20*np.sin(np.arange(68))])
            entries.append({'image':str(path.relative_to(root)),'points':points.tolist(),'box':[0,0,178,178]})
        for name,subset in [('train',entries[:2]),('val',entries[2:])]:
            (root/f'{name}.json').write_text(json.dumps(subset),encoding='utf-8')
        train_landmarks(SimpleNamespace(root=str(root),val_root=None,train=str(root/'train.json'),val=str(root/'val.json'),
            output=str(root/'landmark-run'),epochs=1,batch_size=2,seed=42,device='cpu',lr=.001))
        first=root/entries[0]['image']
        infer_landmarks(SimpleNamespace(checkpoint=root/'landmark-run/last.pt',image=first,box=None,output=root/'landmark-result'))
        assert (root/'landmark-result/aligned.png').is_file()
        (root/'celeba').mkdir()
        for index in range(2):
            Image.fromarray(rng.integers(0,256,(178,178,3),dtype=np.uint8)).save(root/'celeba'/f'{index:06d}.jpg')
        (root/'attributes.txt').write_text('2\nBlack_Hair Young\n000000.jpg 1 -1\n000001.jpg -1 1\n')
        (root/'partition.txt').write_text('000000.jpg 0\n000001.jpg 0\n')
        train_gan(SimpleNamespace(root=str(root/'celeba'),labels=str(root/'attributes.txt'),partition=str(root/'partition.txt'),
            attributes=['Black_Hair','Young'],size=64,steps=1,n_critic=1,batch_size=2,width=8,blocks=1,lr=.0001,
            seed=42,device='cpu',output=str(root/'gan'),log_interval=1,save_interval=1))
        generate(SimpleNamespace(checkpoint=root/'gan/last.pt',image=root/'celeba/000000.jpg',targets='0,1',output=root/'translation.png'))
        assert (root/'translation.png').is_file()


def run(output, full=False):
    seed_all(42); torch.set_num_threads(2)
    model=EmbeddingNet(32).train(); head=ArcFace(32,2)
    images=torch.randn(4,3,64,64); labels=torch.tensor([0,0,1,1])
    optimizer=torch.optim.SGD(list(model.parameters())+list(head.parameters()),lr=.001)
    before=head.weight.detach().clone(); features=model(images)
    loss=torch.nn.functional.cross_entropy(head(features,labels),labels)+batch_hard_triplet(features,labels)
    loss.backward(); optimizer.step()
    assert torch.isfinite(loss) and not torch.equal(before,head.weight.detach())
    assert list(PKSampler([0,0,1,1],2,2))
    landmark_model=make_model().train()
    landmarks=landmark_model(images).reshape(4,68,2)
    landmarks.square().mean().backward()
    assert torch.isfinite(landmarks).all()
    generator=Generator(3,width=8,blocks=1); discriminator=Discriminator(3,width=8,depth=3)
    real=torch.randn(2,3,32,32); attributes=torch.tensor([[1.,0.,1.],[0.,1.,0.]])
    fake=generator(real,attributes)
    assert fake.shape==real.shape
    score,classification=discriminator(fake)
    gan_loss=-score.mean()+torch.nn.functional.binary_cross_entropy_with_logits(classification,attributes)
    gan_loss.backward()
    penalty=gradient_penalty(discriminator,real,fake.detach()); penalty.backward()
    assert torch.isfinite(penalty) and any(p.grad is not None for p in generator.parameters())
    if full: end_to_end()
    report={'status':'passed','scope':'SYNTHETIC CODE SMOKE ONLY; random tensors and untrained models',
        'torch':torch.__version__,'checks':['ResNet50/ArcFace forward-backward and optimizer update',
        'P-K sampler','batch-hard triplet gradients','68 landmark regressor gradients','conditional generator shape',
        'discriminator/classifier and WGAN gradient-penalty backward'],
        'benchmark_accuracy':None,'trained_face_model':False,
        'end_to_end_loaders_checkpoints_inference':full,
        'fixture_note':'All temporary synthetic fixtures/checkpoints are deleted after the smoke check.'}
    path=Path(output); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',default='runs/research-smoke.json')
    parser.add_argument('--end-to-end',action='store_true')
    args=parser.parse_args(); run(args.output,args.end_to_end)
