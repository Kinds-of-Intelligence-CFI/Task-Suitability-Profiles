"""
Best-effort memory sampling for a running subprocess and its children.

Inspect's eval logs do not record host RAM or GPU VRAM usage, so this samples
them externally while an `inspect eval` subprocess runs:

- Host RAM: peak summed RSS across the process tree (root PID + descendants),
  via psutil.
- GPU VRAM: peak summed per-process VRAM via `nvidia-smi --query-compute-apps`,
  restricted to PIDs in the process tree.

Everything degrades gracefully: if psutil is not installed RSS is skipped; if
nvidia-smi is missing or fails (e.g. a CPU/MPS laptop), VRAM is skipped. The
monitor never raises into the caller -- a failed sample just yields no data.

Usage:
    with MemoryMonitor(pid) as mon:
        proc.wait()
    print(mon.peak_ram_mb, mon.peak_vram_mb)
"""

import subprocess
import threading
import time

try:
    import psutil
except ImportError:  # psutil is part of the optional `local` extra
    psutil = None


def _nvidia_smi_available() -> bool:
    try:
        subprocess.run(
            ["nvidia-smi", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def _query_vram_by_pid() -> dict[int, float]:
    """Return {pid: used_vram_mb} for all GPU compute processes (best effort)."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {}

    usage: dict[int, float] = {}
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            usage[int(parts[0])] = float(parts[1])
        except ValueError:
            continue
    return usage


class MemoryMonitor:
    """Context manager that samples peak RAM/VRAM of a process tree."""

    def __init__(self, pid: int, interval: float = 0.5):
        self.pid = pid
        self.interval = interval
        self.peak_ram_mb: float | None = None
        self.peak_vram_mb: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._track_ram = psutil is not None
        self._track_vram = _nvidia_smi_available()

    def __enter__(self) -> "MemoryMonitor":
        if self._track_ram or self._track_vram:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 4)

    def _tree_pids(self) -> set[int]:
        if psutil is None:
            return {self.pid}
        try:
            proc = psutil.Process(self.pid)
            pids = {self.pid}
            pids.update(child.pid for child in proc.children(recursive=True))
            return pids
        except psutil.Error:
            return set()

    def _sample_ram(self, pids: set[int]) -> None:
        total = 0.0
        seen = False
        for pid in pids:
            try:
                total += psutil.Process(pid).memory_info().rss
                seen = True
            except psutil.Error:
                continue
        if seen:
            mb = total / (1024 * 1024)
            if self.peak_ram_mb is None or mb > self.peak_ram_mb:
                self.peak_ram_mb = mb

    def _sample_vram(self, pids: set[int]) -> None:
        usage = _query_vram_by_pid()
        total = sum(mb for pid, mb in usage.items() if pid in pids)
        if total > 0:
            if self.peak_vram_mb is None or total > self.peak_vram_mb:
                self.peak_vram_mb = total

    def _run(self) -> None:
        while not self._stop.is_set():
            pids = self._tree_pids()
            if pids:
                if self._track_ram:
                    self._sample_ram(pids)
                if self._track_vram:
                    self._sample_vram(pids)
            self._stop.wait(self.interval)
