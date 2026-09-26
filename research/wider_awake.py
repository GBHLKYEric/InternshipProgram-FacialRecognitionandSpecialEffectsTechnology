"""Temporary Windows SYSTEM_REQUIRED request for this one monitored experiment.

This does not change a power plan or request that the display stay on. Windows
still permits explicit user sleep, including closing the lid. The request is
cleared in finally when the monitor completes/fails/exits or this helper times out.
Reference: https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate
"""
import ctypes
from ctypes import wintypes
import datetime
import json
import os
from pathlib import Path
import sys
import time


def run():
    if sys.platform != 'win32':
        raise RuntimeError('This optional helper is Windows-only')
    root = Path(__file__).resolve().parents[1]
    monitor_pid = json.loads((root/'runs/wider-monitor/lock.json').read_text())['pid']
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    kernel.SetThreadExecutionState.argtypes = [wintypes.DWORD]
    kernel.SetThreadExecutionState.restype = wintypes.DWORD
    monitor = kernel.OpenProcess(0x00100000, False, monitor_pid)  # SYNCHRONIZE only.
    if not monitor:
        raise ctypes.WinError(ctypes.get_last_error())
    destination = root/'reports/wider-awake.json'
    record = {'pid': os.getpid(), 'monitor_pid': monitor_pid,
              'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'flags': 'ES_CONTINUOUS | ES_SYSTEM_REQUIRED', 'display_required': False,
              'power_plan_changed': False, 'manual_sleep_has_priority': True,
              'documentation': 'https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate'}
    active = False
    try:
        previous = kernel.SetThreadExecutionState(0x80000001)
        if not previous:
            raise ctypes.WinError(ctypes.get_last_error())
        active = True
        record.update(status='active', request_return_value=previous)
        destination.write_text(json.dumps(record, indent=2), encoding='utf-8')
        deadline = time.monotonic()+8*3600
        while time.monotonic() < deadline:
            process_state = kernel.WaitForSingleObject(monitor, 0)
            if process_state == 0:
                record['release_reason'] = 'monitor_process_exited'
                break
            if process_state != 258:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                status = json.loads((root/'reports/wider-monitor.json').read_text())['status']
            except (FileNotFoundError, json.JSONDecodeError):
                status = None
            if status in ['completed', 'failed']:
                record['release_reason'] = 'monitor_'+status
                break
            time.sleep(10)
        else:
            record['release_reason'] = 'eight_hour_timeout'
    except Exception as error:
        record.update(status='failed', error=str(error))
        raise
    finally:
        if active:
            cleared = kernel.SetThreadExecutionState(0x80000000)
            record['release_return_value'] = cleared
            record['status'] = 'released' if cleared else 'release_failed'
        kernel.CloseHandle(monitor)
        record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        destination.write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    run()
