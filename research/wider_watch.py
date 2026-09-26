"""One-shot local continuation for an already-running full WIDER experiment.

Wait for the saved training report, then run both independent evaluators once.
This is a child process for this experiment, not a recurring scheduled job.
No network upload or publication is performed. Start only one monitor per run.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def save_status(path, **values):
    values['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(values, indent=2), encoding='utf-8')
    temporary.replace(path)
    print(json.dumps(values), flush=True)


def run(args):
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    lock = root/'runs/wider-monitor/lock.json'
    lock.parent.mkdir(parents=True, exist_ok=True)
    status = root/'reports/wider-monitor.json'
    with lock.open('x', encoding='utf-8') as stream:
        json.dump({'pid': os.getpid(), 'container': args.container, 'purpose': __doc__}, stream)
    try:
        deadline = time.monotonic()+args.timeout_hours*3600
        last_iteration = None
        failures = 0
        while time.monotonic() < deadline:
            progress_path = root/'reports/wider-mmdet-full-progress.json'
            try:
                progress = json.loads(progress_path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                progress = {}
            if progress.get('status') == 'completed':
                break
            # The training container may finish a moment before its report is
            # observed; tolerate two checks before treating that as a failure.
            inspect = subprocess.run([
                'wsl', '-d', args.distro, '-u', 'root', '--', 'docker', 'inspect',
                '--format', '{{.State.Running}}', args.container],
                capture_output=True, text=True, timeout=30)
            if inspect.returncode or inspect.stdout.strip() != 'true':
                failures += 1
                if failures >= 3:
                    raise RuntimeError('Training container stopped without a completed experiment report; inspect its preserved log')
            else:
                failures = 0
            if progress.get('iteration') != last_iteration:
                last_iteration = progress.get('iteration')
                save_status(status, status='waiting_for_existing_training',
                            container=args.container, progress=progress)
            time.sleep(args.interval)
        else:
            raise TimeoutError('Monitor timeout; training is preserved and has not been stopped')
        training_report = json.loads((root/'reports/wider-mmdet-full.json').read_text())
        if training_report.get('status') != 'completed' or training_report.get('prediction_files') != 3226:
            raise ValueError('Completion report does not prove a full 3226-image prediction export')
        work = Path(training_report['training']['output'])
        prediction_dir = work/'wider_predictions'
        report = Path('reports/wider-mmdet-full-official.json')
        reference = Path('reports/wider-mmdet-full-crosscheck.json')
        commands = [
            [sys.executable, '-m', 'research.wider_eval', '--prediction-dir', str(prediction_dir),
             '--output', str(report)],
            [sys.executable, '-m', 'research.wider_reference', '--prediction-dir', str(prediction_dir),
             '--project-report', str(report), '--output', str(reference)]]
        save_status(status, status='evaluating_complete_predictions', prediction_dir=str(prediction_dir))
        for command in commands:
            subprocess.run(command, check=True)
        official = json.loads(report.read_text())
        crosscheck = json.loads(reference.read_text())
        if not crosscheck['all_differences_below_1e-12']:
            raise AssertionError('Independent AP cross-check did not pass')
        save_status(status, status='completed', prediction_dir=str(prediction_dir),
                    training_report='reports/wider-mmdet-full.json',
                    official_report=str(report), reference_report=str(reference),
                    metrics=official['metrics'], crosscheck=crosscheck)
    except Exception as error:
        save_status(status, status='failed', error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container', default='face-vision-wider-full')
    parser.add_argument('--distro', default='Ubuntu-24.04')
    parser.add_argument('--interval', type=float, default=60)
    parser.add_argument('--timeout-hours', type=float, default=8)
    arguments = parser.parse_args()
    if arguments.interval < 10 or arguments.timeout_hours <= 0:
        parser.error('Use an interval >=10 seconds and a positive timeout')
    run(arguments)
