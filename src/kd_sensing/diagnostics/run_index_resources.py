import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from typing import Any


def collect_python_processes() -> list[dict[str, Any]]:
    proc_root = Path("/proc")
    if not proc_root.exists():
        return []
    records: list[dict[str, Any]] = []
    current_pid = os.getpid()
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == current_pid:
            continue
        argv = _read_proc_argv(entry / "cmdline")
        if not argv or not _looks_like_kd_process(argv):
            continue
        public_argv = redact_argv(argv)
        records.append(
            {
                "pid": pid,
                "argv": public_argv,
                "cmdline": shlex.join(public_argv),
                "cwd": _read_proc_cwd(entry),
                "rss_mb": _read_proc_rss_mb(entry / "status"),
                "config_path": _arg_after(argv, "--config") or _arg_after(argv, "-c"),
                "output_dir": _arg_after(argv, "--output-dir"),
                "run_name": _override_value(argv, "output.run_name"),
                "kind": _process_kind(argv),
            }
        )
    return records

def collect_resource_snapshot(processes: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    process_records = processes if processes is not None else collect_python_processes()
    gpu_snapshot = collect_gpu_snapshot()
    process_records = sanitize_process_records(_attach_gpu_usage(process_records, gpu_snapshot))
    return {
        "memory": collect_memory_snapshot(),
        "gpus": gpu_snapshot,
        "processes": process_records,
    }

def collect_memory_snapshot() -> dict[str, Any]:
    meminfo = _read_meminfo()
    if not meminfo:
        return {"available": False, "reason": "/proc/meminfo unavailable"}
    total = meminfo.get("MemTotal")
    available = meminfo.get("MemAvailable")
    swap_total = meminfo.get("SwapTotal")
    swap_free = meminfo.get("SwapFree")
    return {
        "available": True,
        "total_mb": _kb_to_mb(total),
        "available_mb": _kb_to_mb(available),
        "used_mb": _kb_to_mb(total - available) if total is not None and available is not None else None,
        "swap_total_mb": _kb_to_mb(swap_total),
        "swap_used_mb": _kb_to_mb(swap_total - swap_free)
        if swap_total is not None and swap_free is not None
        else None,
    }

def collect_gpu_snapshot() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "reason": "nvidia-smi not found", "devices": [], "processes": []}
    try:
        gpu_rows = subprocess.run(
            [
                executable,
                "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": str(exc), "devices": [], "processes": []}
    if gpu_rows.returncode != 0:
        return {"available": False, "reason": gpu_rows.stderr.strip() or "nvidia-smi failed", "devices": [], "processes": []}
    devices = []
    for row in gpu_rows.stdout.splitlines():
        parts = [part.strip() for part in row.split(",")]
        if len(parts) != 6:
            continue
        devices.append(
            {
                "index": _int_or_none(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "memory_total_mb": _int_or_none(parts[3]),
                "memory_used_mb": _int_or_none(parts[4]),
                "utilization_gpu_percent": _int_or_none(parts[5]),
            }
        )
    gpu_processes = _collect_gpu_processes(executable)
    return {"available": True, "reason": None, "devices": devices, "processes": gpu_processes}

def match_run_process(
    run_dir: Path,
    *,
    config: dict[str, Any],
    processes: list[dict[str, Any]],
) -> dict[str, Any] | None:
    candidates: list[tuple[int, dict[str, Any]]] = []
    run_name = run_dir.name.lower()
    output_run_name = str(config.get("output_run_name") or "").lower()
    experiment = str(config.get("experiment_name") or "").lower()
    for process in processes:
        cmdline = str(process.get("cmdline", "")).lower()
        score = 0
        if str(run_dir).lower() in cmdline:
            score += 6
        if run_name and run_name in cmdline:
            score += 3
        if output_run_name and output_run_name in cmdline:
            score += 3
        if experiment and experiment in cmdline:
            score += 1
        proc_run_name = str(process.get("run_name") or "").lower()
        if proc_run_name and proc_run_name in {run_name, output_run_name}:
            score += 5
        output_dir = process.get("output_dir")
        if output_dir:
            try:
                if run_dir.is_relative_to(Path(output_dir).expanduser().resolve()):
                    score += 2
            except (OSError, ValueError):
                pass
        if score:
            candidates.append((score, dict(process)))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (-item[0], item[1].get("pid", 0)))[0][1]

def _run_resource_summary(process: dict[str, Any] | None) -> dict[str, Any]:
    if process is None:
        return {"process_rss_mb": None, "pid": None, "gpu_indices": []}
    return {
        "process_rss_mb": process.get("rss_mb"),
        "pid": process.get("pid"),
        "gpu_indices": process.get("gpu_indices", []),
    }

def _public_process(process: dict[str, Any] | None) -> dict[str, Any] | None:
    if process is None:
        return None
    process = _sanitize_process_record(process)
    return {
        "pid": process.get("pid"),
        "config_path": process.get("config_path"),
        "run_name": process.get("run_name"),
        "gpu_indices": list(process.get("gpu_indices", [])),
        "kind": process.get("kind"),
        "argv": list(process.get("argv", [])),
        "cmdline": process.get("cmdline"),
    }

def _looks_like_kd_process(command: str | list[str] | tuple[str, ...]) -> bool:
    lower = " ".join(_coerce_argv(command)).lower()
    if "kd_sensing.cli.train" in lower or "kd_sensing.cli.evaluate" in lower:
        return True
    if "kd-sensing-train" in lower or "kd-sensing-evaluate" in lower:
        return True
    return False

def _process_kind(command: str | list[str] | tuple[str, ...]) -> str:
    lower = " ".join(_coerce_argv(command)).lower()
    if "evaluate" in lower:
        return "evaluation"
    return "training"

def _read_proc_argv(path: Path) -> list[str]:
    try:
        raw = path.read_bytes()
    except OSError:
        return []
    return [part for part in raw.decode("utf-8", errors="replace").split("\0") if part]

def _read_proc_cwd(proc_dir: Path) -> str | None:
    try:
        return str((proc_dir / "cwd").resolve())
    except OSError:
        return None

def _read_proc_rss_mb(path: Path) -> float | None:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return round(int(parts[1]) / 1024, 3)
    except (OSError, ValueError):
        return None
    return None

def _arg_after(command: str | list[str] | tuple[str, ...], flag: str) -> str | None:
    parts = _coerce_argv(command)
    for index, part in enumerate(parts):
        if part == flag and index + 1 < len(parts):
            return parts[index + 1]
        if part.startswith(flag + "="):
            return part.split("=", 1)[1]
    return None

def _override_value(command: str | list[str] | tuple[str, ...], key: str) -> str | None:
    for part in _coerce_argv(command):
        if part.startswith(key + "="):
            return part.split("=", 1)[1]
    return None


def redact_argv(command: str | list[str] | tuple[str, ...]) -> list[str]:
    argv = _coerce_argv(command)
    result: list[str] = []
    redact_next = False
    for part in argv:
        if redact_next:
            result.append("<redacted>")
            redact_next = False
            continue
        if part.startswith("--") and "=" not in part and _is_sensitive_key(part[2:]):
            result.append(part)
            redact_next = True
            continue
        key, separator, value = part.partition("=")
        if separator and _is_sensitive_key(key.lstrip("-")):
            result.append(f"{key}=<redacted>")
            continue
        result.append(re.sub(r"(?i)(https?://)[^/@\s]+@", r"\1<redacted>@", part))
    return result


def redact_command(command: Any) -> str | None:
    if command in (None, ""):
        return None
    return shlex.join(redact_argv(command))


def _sanitize_process_record(process: dict[str, Any]) -> dict[str, Any]:
    public_fields = {
        "pid",
        "cwd",
        "rss_mb",
        "config_path",
        "output_dir",
        "run_name",
        "kind",
        "gpu_usage",
        "gpu_indices",
    }
    copy = {key: value for key, value in process.items() if key in public_fields}
    for key in ("cwd", "config_path", "output_dir", "run_name"):
        if copy.get(key) is not None:
            copy[key] = _redact_uri_userinfo(str(copy[key]))
    argv = redact_argv(process.get("argv") or process.get("cmdline") or [])
    copy["argv"] = argv
    copy["cmdline"] = shlex.join(argv)
    return copy


def sanitize_process_records(processes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_sanitize_process_record(item) for item in processes]


def _coerce_argv(command: Any) -> list[str]:
    if command in (None, ""):
        return []
    if isinstance(command, (list, tuple)):
        return [str(item) for item in command]
    try:
        return shlex.split(str(command))
    except ValueError:
        return str(command).split()


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return bool(re.search(r"(?:^|[._])(password|passwd|token|secret|credential|api_key)(?:$|[._])", normalized))


def _redact_uri_userinfo(value: str) -> str:
    return re.sub(r"(?i)(https?://)[^/@\s]+@", r"\1<redacted>@", value)

def _read_meminfo() -> dict[str, int]:
    path = Path("/proc/meminfo")
    if not path.exists():
        return {}
    values: dict[str, int] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            key, _, rest = line.partition(":")
            amount = rest.strip().split()
            if amount:
                values[key] = int(amount[0])
    except (OSError, ValueError):
        return {}
    return values

def _collect_gpu_processes(executable: str) -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            [
                executable,
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    records = []
    for row in result.stdout.splitlines():
        parts = [part.strip() for part in row.split(",")]
        if len(parts) != 3:
            continue
        records.append({"pid": _int_or_none(parts[0]), "gpu_uuid": parts[1], "memory_used_mb": _int_or_none(parts[2])})
    return records

def _attach_gpu_usage(processes: list[dict[str, Any]], gpu_snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    if not processes or not gpu_snapshot.get("available"):
        return processes
    uuid_to_index = {
        device.get("uuid"): device.get("index")
        for device in gpu_snapshot.get("devices", [])
        if device.get("uuid") is not None
    }
    usage_by_pid: dict[int, list[dict[str, Any]]] = {}
    for item in gpu_snapshot.get("processes", []):
        pid = item.get("pid")
        if pid is None:
            continue
        usage = dict(item)
        usage["gpu_index"] = uuid_to_index.get(item.get("gpu_uuid"))
        usage_by_pid.setdefault(int(pid), []).append(usage)
    enriched = []
    for process in processes:
        copy = dict(process)
        usage = usage_by_pid.get(int(copy.get("pid", -1)), [])
        copy["gpu_usage"] = usage
        copy["gpu_indices"] = [item.get("gpu_index") for item in usage if item.get("gpu_index") is not None]
        enriched.append(copy)
    return enriched

def _kb_to_mb(value: int | None) -> float | None:
    return round(value / 1024, 3) if value is not None else None

def _int_or_none(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None

def _empty_resources(processes: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "memory": {"available": False, "reason": "resource snapshot disabled"},
        "gpus": {"available": False, "reason": "resource snapshot disabled", "devices": [], "processes": []},
        "processes": sanitize_process_records(processes or []),
    }
