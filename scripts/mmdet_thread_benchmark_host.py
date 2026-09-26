"""One-shot Windows supervisor: pause, measure, and always unpause real training.

Run this detached so a chat interruption does not interrupt its finally block.
The disposable benchmark container has a three-minute timeout and is stopped
before the original container is unpaused on every success/failure path.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time
import traceback


def main(args):
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    prefix = ['wsl', '-d', args.distro, '-u', 'root', '--', 'docker']
    benchmark = 'face-vision-wider-thread-bench'
    record = {'status': 'starting', 'main_container': args.container,
              'benchmark_container': benchmark,
              'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    destination = root/'reports/wider-thread-benchmark-supervisor.json'
    paused = False
    pause_start = None
    try:
        subprocess.run(prefix+['pause', args.container], check=True, timeout=30)
        paused = True
        pause_start = time.perf_counter()
        checkpoint = (root/'runs/wider-mmdet-full-stable/last_checkpoint').read_text().strip()
        # MMEngine only updates last_checkpoint after saving the complete file.
        if not checkpoint.startswith('/project/runs/wider-mmdet-full-stable/iter_') or not checkpoint.endswith('.pth'):
            raise ValueError('Unexpected checkpoint path; keep this supervisor scoped to this experiment')
        record.update(status='main_paused_for_measurement', checkpoint=checkpoint)
        destination.write_text(json.dumps(record, indent=2))
        command = prefix+['run', '--rm', '--name', benchmark, '--cpus', '8',
                          '-e', 'OMP_NUM_THREADS=8', '-e', 'MKL_NUM_THREADS=8', '-e', 'OPENBLAS_NUM_THREADS=1',
                          '-v', args.linux_project+':/project', 'face-vision-mmdet:3.3',
                          'python', 'scripts/mmdet_thread_benchmark.py', '--checkpoint', checkpoint]
        subprocess.run(command, check=True, timeout=180)
        record['status'] = 'measurement_completed'
    except Exception as error:
        record.update(status='failed', error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        # Harmless "No such container" is expected after a successful --rm run.
        try:
            cleanup = subprocess.run(prefix+['stop', '--time', '5', benchmark],
                                     capture_output=True, text=True, timeout=20)
            record['benchmark_cleanup_returncode'] = cleanup.returncode
        except Exception as cleanup_error:
            record['benchmark_cleanup_error'] = str(cleanup_error)
        if paused:
            try:
                restored = subprocess.run(prefix+['unpause', args.container], capture_output=True, text=True, timeout=30)
                record['unpause_returncode'] = restored.returncode
                record['unpause_output'] = restored.stdout+restored.stderr
                if restored.returncode:
                    record['status'] = 'requires_unpause_repair'
            except Exception as restore_error:
                record.update(status='requires_unpause_repair', unpause_error=str(restore_error))
            record['main_process_paused_seconds'] = time.perf_counter()-pause_start
        record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        destination.write_text(json.dumps(record, indent=2))
        print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container', default='face-vision-wider-full')
    parser.add_argument('--distro', default='Ubuntu-24.04')
    parser.add_argument('--linux-project', required=True)
    main(parser.parse_args())
