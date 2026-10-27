"""Linux process identities and durable locks for medical queue handoffs.

All signals target a single process, never its process group.  An identity is
the pair (pid, /proc stat starttime), not a PID alone.  No torch dependency is
needed by the supervisor while experiment workers occupy the GPUs.
"""
from contextlib import contextmanager
import ctypes
import errno
import fcntl
import json
import os
from pathlib import Path
import signal
import tempfile
import time


def _stat(pid):
    try:
        raw = (Path("/proc") / str(pid) / "stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    # comm is parenthesized and may itself contain spaces and ')' characters.
    fields = raw.rsplit(")", 1)[1].split()
    return {"pid": pid, "state": fields[0], "ppid": int(fields[1]), "starttime": int(fields[19])}


def process_identity(pid):
    """Return an identity and readable metadata, or None for a missing PID."""
    pid = int(pid)
    if pid <= 0:
        return None
    first = _stat(pid)
    if first is None:
        return None
    proc = Path("/proc") / str(pid)
    try:
        cmdline = [part.decode(errors="replace") for part in (proc / "cmdline").read_bytes().split(b"\0") if part]
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        cmdline = []
    try:
        cwd = os.readlink(proc / "cwd")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        cwd = None
    current = _stat(pid)
    if current is None or current["starttime"] != first["starttime"]:
        return None
    return {**current, "cmdline": cmdline, "cwd": cwd}


def same_process(identity):
    """Whether this PID still refers to the recorded process, including zombies."""
    if identity is None:
        return False
    current = _stat(int(identity["pid"]))
    return current is not None and current["starttime"] == int(identity["starttime"])


def process_running(identity):
    """Stopped processes are live; zombie and dead processes are already exited."""
    if identity is None:
        return False
    current = _stat(int(identity["pid"]))
    return current is not None and current["starttime"] == int(identity["starttime"]) and current["state"] not in {"Z", "X", "x"}


def children_of(identity):
    """Find direct children created by any thread of the recorded process."""
    if not same_process(identity):
        return []
    pid = int(identity["pid"])
    children = set()
    for path in (Path("/proc") / str(pid) / "task").glob("*/children"):
        try:
            children.update(int(value) for value in path.read_text().split())
        except (FileNotFoundError, ProcessLookupError):
            continue
    result = []
    for child_pid in sorted(children):
        child = process_identity(child_pid)
        if child is not None and child["ppid"] == pid:
            result.append(child)
    return result if same_process(identity) else []


def _signal_target(identity):
    pid = int(identity["pid"])
    if pid <= 1 or pid == os.getpid():
        raise ValueError("refusing to signal self or a system PID")
    current = _stat(pid)
    if current is None:
        return None
    if current["starttime"] != int(identity["starttime"]):
        raise ValueError(f"PID {pid} no longer belongs to the recorded process")
    if current["state"] in {"Z", "X", "x"}:
        return None
    return current


def _pidfd_functions():
    if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
        return os.pidfd_open, signal.pidfd_send_signal
    # Some conda Python builds omit these wrappers despite a modern Linux/glibc.
    # Calling the actual libc functions preserves the kernel identity guarantee.
    libc = ctypes.CDLL(None, use_errno=True)
    if not hasattr(libc, "pidfd_open") or not hasattr(libc, "pidfd_send_signal"):
        raise RuntimeError("safe signaling requires pidfd support in Python or libc")
    libc.pidfd_open.argtypes = [ctypes.c_int, ctypes.c_uint]
    libc.pidfd_open.restype = ctypes.c_int
    libc.pidfd_send_signal.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
    libc.pidfd_send_signal.restype = ctypes.c_int

    def checked(result):
        if result < 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))
        return result

    def open_pidfd(pid, flags):
        return checked(libc.pidfd_open(pid, flags))

    def send_pidfd(fd, sig, info, flags):
        return checked(libc.pidfd_send_signal(fd, sig, info, flags))

    return open_pidfd, send_pidfd


def signal_process(identity, sig):
    """Signal only the verified process; reject PID reuse and dangerous targets.

    pidfds pin the kernel process identity.  A system without working pidfds is
    rejected; PID checks around os.kill cannot eliminate PID reuse races.
    """
    current = _signal_target(identity)
    if current is None:
        return False
    pid = current["pid"]
    open_pidfd, send_pidfd = _pidfd_functions()
    try:
        pidfd = open_pidfd(pid, 0)
    except ProcessLookupError:
        return False
    except OSError as exc:
        if exc.errno in {errno.ENOSYS, errno.EINVAL}:
            raise RuntimeError("safe signaling requires kernel pidfd support") from exc
        raise
    try:
        # Recheck after opening the pidfd: the PID could have been recycled in
        # between the first identity check and opening the descriptor.
        if _signal_target(identity) is None:
            return False
        try:
            send_pidfd(pidfd, sig, None, 0)
        except ProcessLookupError:
            return False
        except OSError as exc:
            if exc.errno == errno.ENOSYS:
                raise RuntimeError("safe signaling requires kernel pidfd_send_signal support") from exc
            raise
        return True
    finally:
        os.close(pidfd)


def stop_process(identity, timeout=5):
    """Stop a single process and confirm T/t; its workers continue normally."""
    if timeout < 0:
        raise ValueError("timeout must be nonnegative")
    if not signal_process(identity, signal.SIGSTOP):
        return False
    deadline = time.monotonic() + timeout
    while True:
        current = _signal_target(identity)
        if current is None:
            return False
        if current["state"] in {"T", "t"}:
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"PID {identity['pid']} did not stop within {timeout}s")
        time.sleep(min(0.05, remaining))


def wait_exited(identity, timeout):
    """Wait at most timeout seconds for this identity to exit, including zombie."""
    if timeout < 0:
        raise ValueError("timeout must be nonnegative")
    deadline = time.monotonic() + timeout
    while process_running(identity):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.05, remaining))
    return True


@contextmanager
def exclusive_lock(path):
    """Nonblocking flock; preserve the lock file and all its existing contents."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield handle
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def locks_available(paths):
    """Probe all locks simultaneously, closing every descriptor on every path.

    This is a momentary observation; callers must still acquire the actual GPU
    locks before launching work. Duplicate paths or inode aliases are errors.
    """
    paths = [Path(path) for path in paths]
    canonical = [path.resolve() for path in paths]
    if len(set(canonical)) != len(canonical):
        raise ValueError("duplicate lock paths")
    handles = []
    inodes = set()
    try:
        # Open and validate the entire set before acquiring any lock so inode
        # aliases are reported as bad configuration instead of lock contention.
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a+")
            handles.append(handle)
            stat = os.fstat(handle.fileno())
            inode = stat.st_dev, stat.st_ino
            if inode in inodes:
                raise ValueError("duplicate lock inodes")
            inodes.add(inode)
        for handle in handles:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
        return True
    finally:
        for handle in handles:
            handle.close()


def atomic_write_json(path, obj):
    """Durably replace JSON; serialization failure leaves the old state intact."""
    payload = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
