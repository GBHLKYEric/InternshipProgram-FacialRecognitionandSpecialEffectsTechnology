"""Educational BytePS/ByteNN architecture simulations, NOT either vendor SDK.

python -m research.ecosystem_simulation parameter-server
python -m research.ecosystem_simulation inference-engine --model runs/arcface-pilot/optimized/embedding.onnx --images data/lfw-pilot-train
"""
import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def worker(pipe, images, labels):
    """A worker owns a shard; it returns mean gradients, never updates weights."""
    torch.set_num_threads(1)
    model = nn.Linear(4, 2)
    x, y = torch.tensor(images), torch.tensor(labels)
    try:
        while (weights := pipe.recv()) is not None:
            model.load_state_dict({name: torch.tensor(value) for name, value in weights.items()})
            model.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            pipe.send({'count': len(y), 'loss': loss.item(),
                       'gradients': [p.grad.numpy().copy() for p in model.parameters()]})
    finally:
        pipe.close()


def parameter_server():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    # ponytail: local Pipes teach synchronous aggregation; no network, GPU or
    # scheduler claims. Real multi-node deployment requires the actual framework.
    x, y = torch.randn(8, 4), torch.tensor([0, 1, 0, 1, 1, 0, 1, 0])
    server = nn.Linear(4, 2)
    reference = copy.deepcopy(server)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.1)
    ctx, processes, pipes = mp.get_context('spawn'), [], []
    for start, end in [(0, 3), (3, 8)]:
        parent, child = ctx.Pipe()
        process = ctx.Process(target=worker, args=(child, x[start:end].numpy(), y[start:end].numpy()))
        process.start()
        child.close()
        processes.append(process)
        pipes.append(parent)
    start_time, history, payload_bytes = time.perf_counter(), [], 0
    try:
        for step in range(5):
            weights = {name: value.detach().numpy().copy() for name, value in server.state_dict().items()}
            for pipe in pipes:
                pipe.send(weights)
                payload_bytes += len(pickle.dumps(weights, protocol=4))
            updates = []
            for pipe in pipes:
                if not pipe.poll(90):
                    raise TimeoutError('Worker failed to return gradients in 90 seconds')
                update = pipe.recv()
                updates.append(update)
                payload_bytes += len(pickle.dumps(update, protocol=4))
            total = sum(update['count'] for update in updates)
            with torch.no_grad():
                for index, parameter in enumerate(server.parameters()):
                    gradient = sum(torch.tensor(update['gradients'][index]) * (update['count'] / total)
                                   for update in updates)
                    parameter.sub_(.1 * gradient)
            reference_optimizer.zero_grad()
            reference_loss = F.cross_entropy(reference(x), y)
            reference_loss.backward()
            reference_optimizer.step()
            max_error = max((a-b).abs().max().item() for a, b in zip(server.parameters(), reference.parameters()))
            if max_error > 1e-6:
                raise AssertionError(f'Weighted distributed SGD differs from full-batch SGD: {max_error}')
            history.append({'step': step + 1,
                            'distributed_loss': sum(u['loss'] * u['count'] / total for u in updates),
                            'reference_loss': reference_loss.item(), 'parameter_max_abs_error': max_error})
    finally:
        for pipe in pipes:
            try:
                pipe.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
            pipe.close()
        for process in processes:
            process.join(5)
            if process.is_alive():
                process.terminate()
                process.join()
    if any(p.exitcode != 0 for p in processes):
        raise RuntimeError('One or more workers exited unsuccessfully')
    return {'experiment': 'BytePS concept simulation using Python multiprocessing and PyTorch',
            'actual_byteps_sdk_used': False, 'workers': 2, 'shard_sizes': [3, 5],
            'steps': 5, 'torch': torch.__version__, 'local_cpu_only': True,
            'data': 'Seed-42 synthetic 8x4 teaching vectors; no face accuracy result',
            'elapsed_seconds_including_worker_startup': time.perf_counter()-start_time,
            'serialized_payload_bytes_protocol4_excluding_transport_headers': payload_bytes,
            'equivalence_tolerance': 1e-6, 'history': history,
            'conclusion': 'Weighted gradient aggregation matches full-batch SGD; no speedup claim.'}


def inference_engine(model, images, repeats):
    import onnxruntime as ort
    from PIL import Image
    from research.recognition import face_transform
    from research.optimize import latency_ms
    model = Path(model)
    paths = sorted(p for p in Path(images).rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})[:8]
    if not paths or repeats < 3:
        raise ValueError('Need representative aligned images and at least 3 timing repeats')
    tensors = []
    for path in paths:
        with Image.open(path) as image:
            tensors.append(face_transform()(image.convert('RGB')))
    batch = torch.stack(tensors).numpy()
    sessions = []
    for level in [ort.GraphOptimizationLevel.ORT_DISABLE_ALL, ort.GraphOptimizationLevel.ORT_ENABLE_ALL]:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.graph_optimization_level = level
        sessions.append(ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider']))
    results = []
    for size in sorted({1, len(batch)}):
        inputs = {'images': batch[:size]}
        reference, optimized = [session.run(None, inputs)[0] for session in sessions]
        np.testing.assert_allclose(reference, optimized, rtol=1e-3, atol=1e-5)
        timings = [latency_ms(lambda s=s: s.run(None, inputs), repeats=repeats) for s in sessions]
        results.append({'batch_size': size, 'unoptimized': timings[0], 'optimized': timings[1],
                        'speed_ratio_unoptimized_over_optimized': timings[0]['median_ms']/timings[1]['median_ms'],
                        'embedding_max_abs_error': float(np.max(np.abs(reference-optimized)))})
    with model.open('rb') as file:
        digest = hashlib.file_digest(file, 'sha256').hexdigest()
    return {'experiment': 'ByteNN concept simulation using ONNX Runtime graph optimization',
            'actual_bytenn_sdk_used': False, 'onnxruntime': ort.__version__,
            'model_sha256': digest, 'model': model.as_posix(), 'input': 'RGB aligned faces, 112x112, [-1,1]',
            'images': len(paths), 'provider': 'CPUExecutionProvider', 'intra_op_threads': 2,
            'warmup': 3, 'results': results,
            'limitations': 'Desktop CPU only. Numerical consistency is not face accuracy or mobile validation. '
                           'Other project jobs may contend for CPU; compare within this run only.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=['parameter-server', 'inference-engine'])
    parser.add_argument('--model', default='runs/arcface-pilot/optimized/embedding.onnx')
    parser.add_argument('--images', default='data/lfw-pilot-train')
    parser.add_argument('--repeats', type=int, default=15)
    args = parser.parse_args()
    result = parameter_server() if args.experiment == 'parameter-server' else inference_engine(args.model, args.images, args.repeats)
    destination = Path('reports') / (args.experiment + '-simulation.json')
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
