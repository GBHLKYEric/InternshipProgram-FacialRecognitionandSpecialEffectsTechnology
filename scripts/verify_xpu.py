"""Validate real Intel XPU autograd and the project's StarGAN training operations.

Run using the isolated work/xpu-env interpreter. Uses synthetic inputs, never
downloads a dataset, and does not alter drivers or the project's CPU environment.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import time
import traceback

import torch
import torchvision

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from research.stargan import Generator, Discriminator, gradient_penalty


def timed(operation, device, warmup=2, repetitions=5):
    def sync():
        if device == 'xpu':
            torch.xpu.synchronize()
    for _ in range(warmup):
        operation()
    sync()
    durations = []
    for _ in range(repetitions):
        sync()
        start = time.perf_counter()
        operation()
        sync()
        durations.append(1000 * (time.perf_counter() - start))
    return {'median_ms': statistics.median(durations), 'samples_ms': durations, 'warmup': warmup, 'synchronization': 'torch.xpu.synchronize before/after each measured operation' if device == 'xpu' else 'CPU calls synchronous'}


def generator_benchmark(device):
    torch.manual_seed(42)
    generator = Generator(5, width=64, blocks=6, running_stats=True).to(device).eval()
    image = torch.randn(2, 3, 128, 128, device=device)
    attrs = torch.ones(2, 5, device=device)
    with torch.inference_mode():
        output = generator(image, attrs)
        assert torch.isfinite(output).all()
        result = timed(lambda: generator(image, attrs), device, repetitions=3)
    result.update({'architecture': 'Project Generator, width64, six residual blocks, five attributes', 'input': [2, 3, 128, 128], 'dtype': 'float32', 'includes': 'forward only, model/input already on device', 'output_shape': list(output.shape)})
    return result


def train_smoke(device):
    torch.manual_seed(42)
    generator = Generator(5, width=16, blocks=2).to(device)
    discriminator = Discriminator(5, width=16, depth=3, image_size=64).to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    real = torch.randn(2, 3, 64, 64, device=device)
    attrs = torch.ones(2, 5, device=device)
    final_losses = {}

    def step():
        d_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            fake = generator(real, attrs)
        real_score, real_class = discriminator(real)
        fake_score, _ = discriminator(fake)
        d_loss = fake_score.mean() - real_score.mean() + torch.nn.functional.binary_cross_entropy_with_logits(real_class, attrs) + 10 * gradient_penalty(discriminator, real, fake)
        assert torch.isfinite(d_loss)
        d_loss.backward()
        d_optimizer.step()
        for parameter in discriminator.parameters():
            parameter.requires_grad_(False)
        g_optimizer.zero_grad(set_to_none=True)
        fake = generator(real, attrs)
        score, classification = discriminator(fake)
        g_loss = -score.mean() + torch.nn.functional.binary_cross_entropy_with_logits(classification, attrs) + 10 * torch.nn.functional.l1_loss(generator(fake, attrs), real)
        assert torch.isfinite(g_loss)
        g_loss.backward()
        g_optimizer.step()
        for parameter in discriminator.parameters():
            parameter.requires_grad_(True)
        final_losses.update(d_loss=d_loss.detach().item(), g_loss=g_loss.detach().item())

    result = timed(step, device, warmup=1, repetitions=3)
    result.update({'model': 'Small width16/two-block G and depth3 D', 'input': [2, 3, 64, 64], 'dtype': 'float32', 'validated': ['Conv2d', 'ConvTranspose2d', 'InstanceNorm2d', 'Adam', 'WGAN-GP create_graph double backward', 'generator reconstruction backward'], 'final_losses': final_losses})
    return result


def main():
    torch.set_num_threads(4)
    report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version, 'executable': sys.executable, 'torch': torch.__version__, 'torchvision': torchvision.__version__, 'official_wheel_index': 'https://download.pytorch.org/whl/xpu', 'xpu_available': torch.xpu.is_available(), 'cpu_threads': torch.get_num_threads(), 'limitations': ['Synthetic small compatibility checks are not dataset training or quality evaluation.', 'Concurrent work and power/thermal state were not controlled; timings apply only to the recorded shapes and operations.', 'MMDetection 3.3 uses its separate older CPU environment; successful XPU PyTorch does not migrate MMCV native operators automatically.'], 'checks': {}}
    try:
        assert report['xpu_available'], 'Official XPU wheel cannot initialize this installed driver/device.'
        report['device_name'] = torch.xpu.get_device_name(0)
        properties = torch.xpu.get_device_properties(0)
        report['device_properties'] = {name: getattr(properties, name) for name in ['total_memory', 'max_compute_units', 'driver_version', 'is_integrated_gpu', 'has_fp16', 'has_fp64']}
        vector = torch.arange(1, 17, dtype=torch.float32, device='xpu', requires_grad=True)
        vector.square().sum().backward()
        torch.xpu.synchronize()
        assert torch.equal(vector.grad.cpu(), 2 * torch.arange(1, 17, dtype=torch.float32))
        report['checks']['xpu_transfer_autograd'] = True
        try:
            indices = torchvision.ops.nms(torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device='xpu'), torch.tensor([.9, .8], device='xpu'), .5)
            torch.xpu.synchronize()
            report['checks']['torchvision_nms_xpu'] = indices.cpu().tolist() == [0]
        except Exception as error:
            report['checks']['torchvision_nms_xpu'] = {'error': str(error)}
        for name, operation in [('stargan_training', train_smoke), ('stargan_generator_inference', generator_benchmark)]:
            report['checks'][name] = {}
            for device in ['xpu', 'cpu']:
                try:
                    report['checks'][name][device] = operation(device)
                except Exception:
                    report['checks'][name][device] = {'error': traceback.format_exc()}
            values = report['checks'][name]
            if all('median_ms' in values[d] for d in ['xpu', 'cpu']):
                values['cpu_median_divided_by_xpu_median'] = values['cpu']['median_ms'] / values['xpu']['median_ms']
        report['status'] = 'xpu_available_and_autograd_verified'
    except Exception:
        report['status'] = 'xpu_initialization_or_autograd_failed'
        report['error'] = traceback.format_exc()
    destination = PROJECT / 'reports/xpu-environment.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if report['status'].endswith('failed'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
