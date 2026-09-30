"""Resumable, checksum-verified upload of the 97 raw FR-FCM-Z8P9 CSV files.

Password is requested interactively and never written to disk or argv.
Each remote file is first written to a .part path, verified, then renamed.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import getpass
import hashlib
from pathlib import Path
import shlex
import time

import paramiko


SOURCE = Path(__file__).resolve().parents[1] / "FR-FCM-Z8P9_csv_raw"
DESTINATION = "/ssd1/data/Multi-Organ_Foundation_Model/FR-FCM-Z8P9_csv_raw"
CHUNK_BYTES = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remote_sha256(client: paramiko.SSHClient, path: str) -> str:
    _, stdout, stderr = client.exec_command("sha256sum -- " + shlex.quote(path), timeout=1200)
    value = stdout.read().decode("ascii", errors="replace").strip().split()
    error = stderr.read().decode("utf-8", errors="replace").strip()
    if stdout.channel.recv_exit_status() != 0 or not value:
        raise RuntimeError(f"Remote checksum failed for {path}: {error}")
    return value[0]


def connect(host: str, port: int, user: str, password: str) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(host, port=port, username=user, password=password,
                   timeout=20, auth_timeout=20, banner_timeout=20,
                   compress=True, look_for_keys=False, allow_agent=False)
    return client


def one_file(client: paramiko.SSHClient, source: Path) -> str:
    target = DESTINATION + "/" + source.name
    partial = target + ".part"
    local_size = source.stat().st_size
    if local_size <= 0:
        raise ValueError(f"Empty source file: {source}")
    local_digest = sha256_file(source)
    sftp = client.open_sftp()
    try:
        try:
            existing = sftp.stat(target)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if existing.st_size != local_size or remote_sha256(client, target) != local_digest:
                raise FileExistsError(f"Remote final file differs; refusing overwrite: {target}")
            return "verified-existing"

        try:
            offset = sftp.stat(partial).st_size
        except FileNotFoundError:
            offset = 0
        if offset > local_size:
            raise ValueError(f"Remote partial file is larger than source: {partial}")
        mode = "r+b" if offset else "wb"
        with source.open("rb") as local, sftp.open(partial, mode, bufsize=CHUNK_BYTES) as remote:
            local.seek(offset)
            remote.seek(offset)
            remote.set_pipelined(True)
            for chunk in iter(lambda: local.read(CHUNK_BYTES), b""):
                remote.write(chunk)
        if sftp.stat(partial).st_size != local_size:
            raise IOError(f"Remote file size mismatch after upload: {partial}")
        if remote_sha256(client, partial) != local_digest:
            raise IOError(f"Remote SHA-256 mismatch after upload: {partial}")
        sftp.rename(partial, target)
        return "uploaded" if offset == 0 else f"resumed-from-{offset}"
    finally:
        sftp.close()


def upload_group(files: list[Path], host: str, port: int, user: str,
                 password: str, worker: int) -> list[tuple[str, str]]:
    completed: list[tuple[str, str]] = []
    client: paramiko.SSHClient | None = None
    try:
        for index, source in enumerate(files, 1):
            for attempt in range(1, 7):
                try:
                    if client is None:
                        client = connect(host, port, user, password)
                    status = one_file(client, source)
                    completed.append((source.name, status))
                    print(f"worker={worker} {index}/{len(files)} {source.name} {status}", flush=True)
                    break
                except (OSError, EOFError, paramiko.SSHException) as exc:
                    if client is not None:
                        client.close()
                        client = None
                    if attempt == 6:
                        raise
                    delay = min(20, 2 ** attempt)
                    print(f"worker={worker} retry={attempt} {source.name}: {exc}; wait={delay}s", flush=True)
                    time.sleep(delay)
    finally:
        if client is not None:
            client.close()
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="frp-say.com")
    parser.add_argument("--port", type=int, default=44594)
    parser.add_argument("--user", default="user")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be 1..4")
    if not SOURCE.is_dir():
        parser.error(f"Missing source directory: {SOURCE}")
    files = sorted((path for path in SOURCE.iterdir() if path.is_file() and path.suffix.lower() == ".csv"),
                   key=lambda path: path.stat().st_size, reverse=True)
    if len(files) != 97:
        parser.error(f"Expected 97 CSV files, found {len(files)}")
    total_bytes = sum(path.stat().st_size for path in files)
    password = getpass.getpass(f"SSH password for {args.user}@{args.host}: ")
    initial = connect(args.host, args.port, args.user, password)
    try:
        sftp = initial.open_sftp()
        try:
            try:
                sftp.stat(DESTINATION)
            except FileNotFoundError:
                sftp.mkdir(DESTINATION)
        finally:
            sftp.close()
    finally:
        initial.close()
    groups: list[list[Path]] = [[] for _ in range(args.workers)]
    group_sizes = [0] * args.workers
    for source in files:
        worker = min(range(args.workers), key=lambda i: group_sizes[i])
        groups[worker].append(source)
        group_sizes[worker] += source.stat().st_size
    print(f"Uploading {len(files)} files, {total_bytes} bytes to {DESTINATION}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(upload_group, group, args.host, args.port,
                               args.user, password, worker)
                   for worker, group in enumerate(groups)]
        results = [future.result() for future in futures]
    uploaded = sum(status != "verified-existing" for group in results for _, status in group)
    print(f"Complete: {sum(map(len, results))}/97 files verified; newly uploaded={uploaded}", flush=True)


if __name__ == "__main__":
    main()
