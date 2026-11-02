"""Supervise only processes created in this run's Linux session, with PID reuse checks."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import time

from common import save_json, utc


def identity(pid):
    try:
        raw = (Path('/proc') / str(pid) / 'stat').read_text()
        fields = raw.rsplit(') ', 1)[1].split()
        return {"pid": int(pid), "state": fields[0], "ppid": int(fields[1]),
                "session": int(fields[3]), "start_ticks": int(fields[19])}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(record):
    actual = identity(record['pid'])
    return actual if actual and actual['start_ticks'] == record['start_ticks'] else None


class OwnedProcess:
    def __init__(self, command, *, cwd, environment, log_path, name="process", shutdown_timeout=30):
        self.cwd, self.name = Path(cwd), name
        self.shutdown_timeout = shutdown_timeout
        self.cwd.mkdir(parents=True, exist_ok=True)
        self.log = Path(log_path).open('a')
        self.closed = False
        try:
            self.process = subprocess.Popen(command, cwd=self.cwd, env=environment,
                                            stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            self.log.close()
            raise
        self.leader = identity(self.process.pid)
        if self.leader is None:
            self.process.wait()
            self.log.close()
            raise RuntimeError('Process vanished before its identity could be recorded')
        self.known = {self.process.pid: self.leader}
        save_json(self.cwd / (name + '-owner.json'), {"created_utc": utc(), "command": command,
                  "leader": self.leader, "ownership_basis": "Popen-created session plus /proc start_ticks"})

    def capture(self):
        # A surviving member holds the session identity; when no known member
        # survives, do not search by the old numeric PID/session identifier.
        members = [same_process(row) for row in self.known.values()]
        if any(row and row['session'] == self.leader['session'] for row in members):
            for path in Path('/proc').iterdir():
                if path.name.isdigit():
                    try:
                        row = identity(int(path.name))
                    except PermissionError:
                        continue
                    if row and row['session'] == self.leader['session'] and row['start_ticks'] >= self.leader['start_ticks']:
                        self.known[row['pid']] = row
        return [row for item in self.known.values() if (row := same_process(item)) and row['state'] != 'Z']

    def poll(self):
        self.capture()
        return self.process.poll()

    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        try:
            while self.poll() is None:
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(self.process.args, timeout)
                time.sleep(0.2)
            return self.process.returncode
        except BaseException:
            self.stop()
            raise

    def stop(self):
        if self.closed:
            return self.cleanup
        events = []
        for sig, duration in [(signal.SIGTERM, self.shutdown_timeout), (signal.SIGKILL, 5)]:
            live = self.capture()
            for row in sorted(live, key=lambda x: x['pid'] == self.leader['pid']):
                if same_process(row):
                    try:
                        os.kill(row['pid'], sig)
                        events.append({"pid": row['pid'], "start_ticks": row['start_ticks'], "signal": sig.name})
                    except ProcessLookupError:
                        pass
            deadline = time.monotonic() + duration
            while time.monotonic() < deadline:
                self.process.poll()
                if not self.capture():
                    break
                time.sleep(0.1)
            if not self.capture():
                break
        alive = self.capture()
        self.process.poll()
        self.log.close()
        self.cleanup = {"checked_utc": utc(), "signals": events, "remaining_owned_processes": alive,
                        "released": not alive, "returncode": self.process.returncode}
        save_json(self.cwd / (self.name + '-cleanup.json'), self.cleanup)
        self.closed = True
        if alive:
            raise RuntimeError('Owned processes still alive; caches must be retained until they are stopped')
        return self.cleanup

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stop()
