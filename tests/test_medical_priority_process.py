"""Exercise scheduling primitives only against disposable child processes."""
import importlib
import importlib.util
import errno
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


def helpers():
    assert importlib.util.find_spec("scripts.medical_priority_process"), "process safety helpers are missing"
    return importlib.import_module("scripts.medical_priority_process")


def wait_until(predicate, timeout=4):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("disposable child did not reach expected state")


def cleanup(proc):
    if proc.poll() is None:
        proc.send_signal(signal.SIGCONT)
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)


def test_process_identity_rejects_reused_pid_and_does_not_signal_sibling(tmp_path):
    victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], cwd=tmp_path)
    sibling = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        h = helpers()
        identity = h.process_identity(victim.pid)
        assert identity["pid"] == victim.pid
        assert identity["ppid"] == os.getpid()
        assert identity["starttime"] > 0
        assert identity["cmdline"][0] == sys.executable
        assert identity["cwd"] == str(tmp_path)
        assert h.same_process(identity)
        assert h.process_running(identity)
        wrong = dict(identity, starttime=identity["starttime"] + 1)
        assert not h.same_process(wrong)
        with pytest.raises(ValueError):
            h.signal_process(wrong, signal.SIGTERM)
        assert victim.poll() is None
        assert h.signal_process(identity, signal.SIGTERM)
        victim.wait(timeout=3)
        assert h.wait_exited(identity, timeout=1)
        assert not h.signal_process(identity, signal.SIGTERM)
        assert sibling.poll() is None
    finally:
        cleanup(victim)
        cleanup(sibling)


def test_parent_stop_preserves_thread_created_worker_and_detects_zombie(tmp_path):
    # The worker belongs to a non-main thread's /proc task children list.
    worker_code = "import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); [(p.write_text(str(i)),time.sleep(.025)) for i in range(50)]"
    parent_code = """
import json, pathlib, subprocess, sys, threading
def launch():
    child = subprocess.Popen([sys.executable, '-c', sys.argv[2], sys.argv[3]])
    pathlib.Path(sys.argv[1]).write_text(str(child.pid))
    child.wait()
t = threading.Thread(target=launch)
t.start()
t.join()
"""
    pidfile, heartbeat = tmp_path / "worker.pid", tmp_path / "heartbeat"
    parent = subprocess.Popen([sys.executable, "-c", parent_code, str(pidfile), worker_code, str(heartbeat)])
    try:
        wait_until(lambda: pidfile.exists() and heartbeat.exists())
        h = helpers()
        identity = h.process_identity(parent.pid)
        worker_pid = int(pidfile.read_text())
        children = h.children_of(identity)
        assert worker_pid in {child["pid"] for child in children}
        worker = next(child for child in children if child["pid"] == worker_pid)
        assert h.stop_process(identity, timeout=2)
        assert h.process_identity(parent.pid)["state"] in {"T", "t"}
        before = heartbeat.read_text()
        wait_until(lambda: heartbeat.read_text() != before)
        assert not h.wait_exited(worker, timeout=0.01)
        wait_until(lambda: h.process_identity(worker_pid)["state"] == "Z")
        assert h.same_process(worker)
        assert not h.process_running(worker)
        assert h.wait_exited(worker, timeout=0.01)
        assert h.process_running(identity)
        assert h.signal_process(identity, signal.SIGCONT)
        parent.wait(timeout=3)
    finally:
        cleanup(parent)


def test_signal_refuses_self_and_system_pid():
    h = helpers()
    with pytest.raises(ValueError):
        h.signal_process(h.process_identity(os.getpid()), signal.SIGSTOP)
    with pytest.raises(ValueError):
        h.signal_process({"pid": 1, "starttime": 0}, signal.SIGSTOP)


@pytest.mark.parametrize("missing", ["python_and_libc", "kernel"])
def test_signal_fails_closed_without_pidfd_support(monkeypatch, missing):
    h = helpers()
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        identity = h.process_identity(proc.pid)
        if missing == "python_and_libc":
            monkeypatch.delattr(os, "pidfd_open", raising=False)
            monkeypatch.delattr(signal, "pidfd_send_signal", raising=False)
            monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: object())
        else:
            def unavailable(*args):
                raise OSError(errno.ENOSYS, "pidfds unavailable in this kernel")
            monkeypatch.setattr(os, "pidfd_open", unavailable, raising=False)
            monkeypatch.setattr(signal, "pidfd_send_signal", lambda *args: None, raising=False)
        with pytest.raises(RuntimeError, match="pidfd"):
            h.signal_process(identity, signal.SIGTERM)
        assert proc.poll() is None
    finally:
        cleanup(proc)


def test_process_stat_parentheses_are_parsed_correctly(tmp_path):
    marker = tmp_path / "ready"
    code = "import ctypes,pathlib,sys,time; ctypes.CDLL(None).prctl(15,b'worker ) name',0,0,0); pathlib.Path(sys.argv[1]).touch(); time.sleep(30)"
    proc = subprocess.Popen([sys.executable, "-c", code, str(marker)])
    try:
        wait_until(marker.exists)
        h = helpers()
        identity = h.process_identity(proc.pid)
        assert identity["ppid"] == os.getpid()
        assert identity["starttime"] > 0
        assert h.same_process(identity)
    finally:
        cleanup(proc)


def test_lock_probe_preserves_existing_inode_and_contents(tmp_path):
    path, ready = tmp_path / "gpu.lock", tmp_path / "ready"
    path.write_text("original owner metadata")
    inode = path.stat().st_ino
    code = "import fcntl,pathlib,sys,time; f=open(sys.argv[1],'a'); fcntl.flock(f,fcntl.LOCK_EX); pathlib.Path(sys.argv[2]).touch(); time.sleep(30)"
    proc = subprocess.Popen([sys.executable, "-c", code, str(path), str(ready)])
    try:
        wait_until(ready.exists)
        h = helpers()
        assert not h.locks_available([path])
        with pytest.raises(BlockingIOError):
            with h.exclusive_lock(path):
                pytest.fail("acquired another process's lock")
        assert path.stat().st_ino == inode
        assert path.read_text() == "original owner metadata"
    finally:
        cleanup(proc)
    assert h.locks_available([path])
    with h.exclusive_lock(path):
        assert not h.locks_available([path])
    assert path.stat().st_ino == inode
    assert path.read_text() == "original owner metadata"


def test_lock_probe_releases_partial_acquisitions_and_rejects_aliases(tmp_path):
    h = helpers()
    first, second = tmp_path / "first.lock", tmp_path / "second.lock"
    with h.exclusive_lock(second):
        assert not h.locks_available([first, second])
        assert h.locks_available([first])
    with pytest.raises(ValueError):
        h.locks_available([first, first.parent / "." / first.name])
    alias = tmp_path / "alias.lock"
    os.link(first, alias)
    with pytest.raises(ValueError):
        h.locks_available([first, alias])


def test_atomic_json_preserves_valid_previous_file_on_serialization_failure(tmp_path):
    h = helpers()
    path = tmp_path / "nested" / "state.json"
    h.atomic_write_json(path, {"phase": "draining", "jobs": [1, 2]})
    assert json.loads(path.read_text()) == {"phase": "draining", "jobs": [1, 2]}
    with pytest.raises(TypeError):
        h.atomic_write_json(path, {"invalid": object()})
    assert json.loads(path.read_text())["phase"] == "draining"
    assert sorted(p.name for p in path.parent.iterdir()) == ["state.json"]
    h.atomic_write_json(path, {"phase": "ready"})
    assert json.loads(path.read_text()) == {"phase": "ready"}
