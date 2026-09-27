"""Build a provenance and elapsed-time audit from this completed WIDER run.

Uses the saved effective configuration of the running process, not the current
launcher source (which was improved while that process continued). The event
intervals are observations; they do not prove zero CPU work throughout standby.
"""
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import shutil


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def instant(text):
    return dt.datetime.fromisoformat(text.replace('Z', '+00:00'))


def main():
    reports = Path('reports')
    training = read(reports / 'wider-mmdet-full.json')
    monitor = read(reports / 'wider-monitor.json')
    awake = read(reports / 'wider-awake.json')
    curves = read(reports / 'wider-mmdet-training-curves.json')
    checkpoint_metadata = read(reports / 'wider-checkpoint-metadata.json')
    container = read(reports / 'wider-mmdet-container.json')
    if training['status'] != 'completed' or monitor['status'] != 'completed':
        raise ValueError('The complete training and independent validation must finish first')
    if awake['status'] != 'released' or not awake['release_return_value']:
        raise ValueError('Temporary Windows power request has not been verified released')
    if curves['status'] != 'completed':
        raise ValueError('Generate the completed training curves first')
    work = Path(training['training']['output'])
    active_config = work / 'effective-config.py'
    if (checkpoint_metadata['iter'] != 6440
            or not checkpoint_metadata['checkpoint_config_equals_saved_config']
            or checkpoint_metadata['checkpoint_config_sha256'] != sha256(active_config)):
        raise ValueError('Final checkpoint iteration and saved configuration must match')
    config_snapshot = reports / 'wider-mmdet-effective-config.py'
    shutil.copyfile(active_config, config_snapshot)
    event_sources = [
        'wider-power-transitions.json',
        'wider-power-transitions-second.json',
        'wider-power-transitions-third.json',
    ]
    intervals = []
    for source in event_sources:
        events = sorted(read(reports / source), key=lambda e: e['time_utc'])
        # This capture contains one explicitly observed entered -> exited pair.
        # Additional isolated events are retained in the source, not inferred.
        pairs = [(left, right) for left, right in zip(events, events[1:])
                 if left['Id'] == 506 and right['Id'] == 507]
        if len(pairs) != 1:
            raise ValueError(f'Expected one observed entered/exited pair in {source}')
        entered, exited = pairs[0]
        intervals.append({
            'source': str(reports / source),
            'entered_utc': entered['time_utc'],
            'exited_utc': exited['time_utc'],
            'entered_message': entered['Message'],
            'exited_message': exited['Message'],
            'observed_interval_seconds': (instant(exited['time_utc']) - instant(entered['time_utc'])).total_seconds(),
        })
    pauses = [read(reports / name)['main_process_paused_seconds'] for name in [
        'wider-thread-benchmark-attempt1-supervisor.json',
        'wider-thread-benchmark-supervisor.json',
    ]]
    environment = reports / 'mmdet-environment-freeze.txt'
    log = (reports / 'mmdet-full-log.txt').read_text(encoding='utf-8-sig')
    python_match = re.search(r'^\s*Python:\s*(.+)$', log, re.MULTILINE)
    coco_summary_lines = [line.strip() for line in log.splitlines()
                          if re.match(r'\s*Average (Precision|Recall)\s', line)]
    packages = [line for line in environment.read_text().splitlines()
                if line.lower().startswith(('torch', 'mmcv', 'mmdet', 'mmengine', 'numpy', 'pycocotools'))]
    result = {
        'status': 'completed',
        'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'container': container,
        'effective_config': {
            'source': str(active_config),
            'published_snapshot': str(config_snapshot),
            'sha256': sha256(active_config),
            'bytes': active_config.stat().st_size,
            'runtime_loop': 'EpochBasedTrainLoop, max_epochs=1, val_interval=2; no val_begin override',
            'scope': 'Exact config saved before this stable training started. Later launcher fixes (resume loop and one-validation scheduling) were not executed by this process.',
        },
        'environment': {'source': str(environment), 'sha256': sha256(environment),
                        'python_from_actual_log': python_match.group(1).strip() if python_match else None,
                        'selected_packages': packages},
        'checkpoint': {key: curves[key] for key in ['final_checkpoint', 'final_checkpoint_sha256', 'final_checkpoint_bytes']},
        'checkpoint_metadata': checkpoint_metadata,
        'training_iterations_completed': 6440,
        'last_loss_logged_iteration': curves['last_iteration'],
        'calendar': {
            'started_at': container['started_at'],
            'independent_evaluation_finished_at': monitor['updated_utc'],
            'elapsed_seconds': (instant(monitor['updated_utc']) - instant(container['started_at'])).total_seconds(),
            'scope': 'Actual UTC interval from Docker process start through both final independent AP evaluations. Includes observed standby and diagnostic pauses; excludes earlier acquisition and failed attempt.',
        },
        'internal_timing': {
            'runner_train_seconds': training['training_seconds'],
            'explicit_second_validation_seconds': training['validation_seconds'],
            'controlled_diagnostic_pause_seconds': sum(pauses),
            'scope': 'runner.train() includes one automatic full validation and diagnostic pauses; explicit validation is the duplicate second pass. Observed WSL perf_counter did not accumulate the long host standby intervals. Neither field is pure training time or user elapsed waiting time.',
        },
        'observed_modern_standby_intervals': intervals,
        'observed_intervals_total_seconds': sum(e['observed_interval_seconds'] for e in intervals),
        'power_event_limitations': 'Only captured adjacent entered/exited pairs are summed. No claim of an exhaustive power history or zero computation at every instant. Numeric reason 16777220 is retained without interpretation.',
        'monitor_attempt1': read(reports / 'wider-monitor-attempt1.json'),
        'power_helper_attempt1': read(reports / 'wider-awake-attempt1.json'),
        'monitor_final': {'status': monitor['status'], 'updated_utc': monitor['updated_utc']},
        'power_helper_final': awake,
        'coco_evaluator': {
            'implementation': 'MMDetection 3.3.0 CocoMetric with pycocotools 2.0.11',
            'proposal_nums_maxDets': [100, 300, 1000],
            'reported_metric_maxDets': {'bbox_mAP': 100, 'bbox_mAP_50': 1000, 'bbox_mAP_75': 1000,
                                       'bbox_mAP_s': 1000, 'bbox_mAP_m': 1000, 'bbox_mAP_l': 1000,
                                       'classwise_face_precision_ap': 1000},
            'use_mp_eval': False,
            'verification': 'Inspected the installed CocoMetric signature in the actual running container; the saved effective config does not override these defaults.',
            'classwise_name_caution': 'coco/face_precision is the classwise face AP in this evaluator, not precision at one confidence operating point.',
            'actual_logged_summaries': coco_summary_lines,
        },
        'evaluation_scope': 'WIDER difficulty AP uses IoU 0.5, official ignore masks and 1000 score thresholds. COCO bbox_mAP averages IoUs 0.50:0.05:0.95 on converted valid boxes, a distinct metric. One epoch at maximum edge 320 does not establish convergence.',
    }
    destination = reports / 'wider-mmdet-audit.json'
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
