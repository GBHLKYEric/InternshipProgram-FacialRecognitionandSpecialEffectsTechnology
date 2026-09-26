"""ResNet50 + ArcFace; train on locally licensed, aligned identity folders.

python -m research.recognition --data data/identities --output runs/arcface
Each identity directory must contain >=2 images. Inputs are RGB faces, 112x112,
normalized to [-1,1]. Random initialization is intentional; LFW is evaluation only.
"""
import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets, models, transforms


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def face_transform(train=False, size=112):
    steps = [transforms.Resize((size, size))]
    if train:
        steps.append(transforms.RandomHorizontalFlip())
    return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize([.5]*3, [.5]*3)])


class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embedding_dim)

    def forward(self, images):
        return F.normalize(self.backbone(images), dim=1)


class ArcFace(nn.Module):
    """Additive angular margin: s*cos(theta_y+m), monotonic fallback near pi."""
    def __init__(self, embedding_dim, classes, margin=.5, scale=64.):
        super().__init__()
        if classes < 2 or not 0 < margin < math.pi/2 or scale <= 0:
            raise ValueError('ArcFace needs >=2 classes, margin in (0,pi/2), scale >0')
        self.weight = nn.Parameter(torch.empty(classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin, self.scale = margin, scale

    def forward(self, features, labels):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight)).clamp(-1+1e-7, 1-1e-7)
        sine = torch.sqrt((1-cosine.square()).clamp_min(1e-7))
        phi = cosine*math.cos(self.margin)-sine*math.sin(self.margin)
        phi = torch.where(cosine > math.cos(math.pi-self.margin), phi,
                          cosine-math.sin(math.pi-self.margin)*self.margin)
        hot = F.one_hot(labels, self.weight.shape[0]).to(cosine.dtype)
        return self.scale * (hot*phi + (1-hot)*cosine)


class PKSampler(Sampler):
    """P different identities, K different images per identity; epoch-seeded."""
    def __init__(self, labels, p=8, k=4, seed=42):
        self.groups = defaultdict(list)
        for index, label in enumerate(labels):
            self.groups[label].append(index)
        if p < 2 or k < 2 or len(self.groups) < p:
            raise ValueError('Need P>=2, K>=2 and at least P identities')
        if any(len(indices) < k for indices in self.groups.values()):
            raise ValueError('Every identity needs at least K distinct images; reduce K or clean data')
        self.p, self.k, self.seed, self.epoch = p, k, seed, 0
        self.batches = max(1, math.ceil(len(labels)/(p*k)))

    def __iter__(self):
        rng = random.Random(self.seed+self.epoch)
        for _ in range(self.batches):
            yield [index for label in rng.sample(list(self.groups), self.p)
                   for index in rng.sample(self.groups[label], self.k)]

    def __len__(self):
        return self.batches


def batch_hard_triplet(features, labels, margin=.2):
    """Hardest positive and negative for each valid anchor in its P-K batch."""
    distances = torch.cdist(features, features)
    same = labels[:, None].eq(labels[None, :])
    positive = same & ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    negative = ~same
    valid = positive.any(1) & negative.any(1)
    if not valid.any():
        raise ValueError('Batch-hard mining needs positive and negative examples per anchor')
    hardest_positive = distances.masked_fill(~positive, float('-inf')).amax(1)
    hardest_negative = distances.masked_fill(~negative, float('inf')).amin(1)
    return F.relu(hardest_positive[valid]-hardest_negative[valid]+margin).mean()


def load_embedding(checkpoint):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if payload.get('model_type') != 'resnet50_arcface':
        raise ValueError('Expected a research.recognition checkpoint')
    model = EmbeddingNet(payload['embedding_dim'])
    model.load_state_dict(payload['model'])
    return model.eval(), payload


def save_curves(history, output):
    """Dependency-free SVG of measured epoch loss and training classification rate."""
    with Path(history).open(encoding='utf-8',newline='') as file: rows=list(csv.DictReader(file))
    if not rows: raise ValueError('No history rows to plot')
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="400" viewBox="0 0 1100 400">',
        '<rect width="1100" height="400" fill="#f8fafc"/>',
        '<text x="40" y="30" font-family="sans-serif" font-size="20">Measured training history (not verification accuracy)</text>']
    for panel,(column,title) in enumerate([('loss','ArcFace + batch-hard loss'),('margin_classifier_train_accuracy','Margin classifier training accuracy')]):
        values=[float(row[column]) for row in rows]; left=55+panel*540; top=75; width=455; height=255
        low=0.; high=max(values)*1.1 if column=='loss' else 1.
        high=max(high,1e-6)
        points=[(left+index*width/max(1,len(rows)-1),top+height-height*(value-low)/(high-low)) for index,value in enumerate(values)]
        parts.extend([f'<text x="{left}" y="55" font-family="sans-serif" font-size="16">{title}</text>',
            f'<path d="M{left},{top} V{top+height} H{left+width}" fill="none" stroke="#64748b"/>'])
        for tick in range(5):
            y=top+height-tick*height/4
            parts.extend([f'<path d="M{left},{y} H{left+width}" stroke="#dbe3eb"/>',
                f'<text x="{left-8}" y="{y+4}" text-anchor="end" font-family="sans-serif" font-size="11">{high*tick/4:.3g}</text>'])
        parts.append('<polyline points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in points)+'" fill="none" stroke="#2563eb" stroke-width="3"/>')
        for row,(x,y),value in zip(rows,points,values):
            parts.extend([f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#2563eb"/>',
                f'<text x="{x:.2f}" y="{y-9:.2f}" text-anchor="middle" font-family="sans-serif" font-size="12">{value:.4f}</text>',
                f'<text x="{x:.2f}" y="{top+height+20}" text-anchor="middle" font-family="sans-serif" font-size="12">{int(row["epoch"])}</text>'])
        parts.append(f'<text x="{left+width/2}" y="385" text-anchor="middle" font-family="sans-serif" font-size="13">Epoch</text>')
    Path(output).write_text('\n'.join(parts+['</svg>']),encoding='utf-8')


def train(args):
    seed_all(args.seed)
    torch.set_num_threads(getattr(args,'threads',2))
    if args.epochs < 1 or args.lr <= 0 or args.triplet_weight < 0:
        raise ValueError('epochs and lr must be positive; triplet weight nonnegative')
    dataset = datasets.ImageFolder(args.data, transform=face_transform(True))
    sampler = PKSampler(dataset.targets, args.p, args.k, args.seed)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers)
    device = torch.device(args.device)
    model, head = EmbeddingNet(args.embedding_dim).to(device), ArcFace(args.embedding_dim, len(dataset.classes)).to(device)
    optimizer = torch.optim.SGD(list(model.parameters())+list(head.parameters()), lr=args.lr, momentum=.9, weight_decay=5e-4)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'classes.json').write_text(json.dumps(dataset.class_to_idx, indent=2), encoding='utf-8')
    (output/'config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    with (output/'history.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'loss', 'margin_classifier_train_accuracy', 'samples'])
        for epoch in range(args.epochs):
            sampler.epoch = epoch
            model.train(); head.train()
            total_loss, correct, count = 0., 0, 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                features = model(images)
                logits = head(features, labels)
                loss = F.cross_entropy(logits, labels) + args.triplet_weight*batch_hard_triplet(features, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite training loss')
                loss.backward()
                nn.utils.clip_grad_norm_(list(model.parameters())+list(head.parameters()), 5.)
                optimizer.step()
                total_loss += float(loss.detach())*len(labels)
                correct += int((logits.argmax(1)==labels).sum())
                count += len(labels)
            row = [epoch+1, total_loss/count, correct/count, count]
            writer.writerow(row); file.flush()
            print(dict(zip(['epoch','loss','train_accuracy','samples'], row)), flush=True)
            torch.save({'model_type':'resnet50_arcface', 'embedding_dim':args.embedding_dim,
                        'model':model.state_dict(), 'head':head.state_dict(),
                        'optimizer':optimizer.state_dict(), 'epoch':epoch+1,
                        'classes':dataset.class_to_idx, 'seed':args.seed}, output/'last.pt')
    save_curves(output/'history.csv',output/'training-curves.svg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default='runs/arcface')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--p', type=int, default=8)
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--embedding-dim', type=int, default=512)
    parser.add_argument('--lr', type=float, default=.05)
    parser.add_argument('--triplet-weight', type=float, default=.1)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    train(parser.parse_args())
