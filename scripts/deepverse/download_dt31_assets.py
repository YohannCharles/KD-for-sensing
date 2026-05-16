#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Asset:
    name: str
    filename: str
    url: str
    size: int
    extract_subdir: str | None
    ranged: bool = True


ASSETS: dict[str, Asset] = {
    "param": Asset(
        name="param",
        filename="param.zip",
        url="https://www.dropbox.com/scl/fo/i0ja9lnuecq96mpsa7dmo/AAZq8V_lHB1JRur5iHyedGQ"
        "?rlkey=2tgnuw1ggpuv526cbv0i3l46k&st=s2q7rp4d&dl=1",
        size=7_745_215,
        extract_subdir="param",
        ranged=False,
    ),
    "wireless": Asset(
        name="wireless",
        filename="wireless.zip",
        url="https://dl.dropboxusercontent.com/scl/fi/pyzmcb5jv3oo9e29nymw8/wireless.zip"
        "?rlkey=b5gbkgd8dng6hpf91rv7za9ix&st=5opqw9lc&dl=1",
        size=318_043_431,
        extract_subdir=None,
    ),
    "lidar": Asset(
        name="lidar",
        filename="lidar.zip",
        url="https://dl.dropboxusercontent.com/scl/fi/v2lyhoez7ce14eqwf7wdy/lidar.zip"
        "?rlkey=05uiwsddg55q2ebwngu6l3i2e&st=u6x0uqvq&dl=1",
        size=10_159_158_017,
        extract_subdir=None,
    ),
    "rgb": Asset(
        name="rgb",
        filename="RGB_images.zip",
        url="https://dl.dropboxusercontent.com/scl/fi/or07xc9dh5x9ehd7wrhq5/RGB_images.zip"
        "?rlkey=podymx17iiwvuvmffiyl2p3qs&st=j34yexb5&dl=1",
        size=22_807_981_179,
        extract_subdir=None,
    ),
}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    download_root = args.download_root.expanduser().resolve()
    scenario_dir = args.scenario_root.expanduser().resolve() / args.scenario
    status_path = args.status.expanduser().resolve()
    selected = resolve_assets(args.assets)

    download_root.mkdir(parents=True, exist_ok=True)
    scenario_dir.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    log(
        f"Starting DeepVerse DT31 assets: assets={','.join(asset.name for asset in selected)}, "
        f"download_root={download_root}, scenario_dir={scenario_dir}"
    )
    write_status(status_path, {"state": "running", "assets": [asset.name for asset in selected]})

    for asset in selected:
        archive = download_root / asset.filename
        download_asset(
            asset=asset,
            archive=archive,
            workers=args.workers,
            chunk_bytes=args.chunk_mib * 1024 * 1024,
            status_path=status_path,
        )
        if args.verify_zip:
            verify_zip(archive)
        if args.extract:
            extract_asset(asset=asset, archive=archive, scenario_dir=scenario_dir, force=args.force_extract)

    ensure_dt31_layout(scenario_dir)

    if args.generate_cache:
        generate_cache(args=args, scenario_dir=scenario_dir)

    write_status(status_path, {"state": "complete", "assets": [asset.name for asset in selected]})
    log("DeepVerse DT31 asset job complete.")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and optionally extract DeepVerse6G-DT31 assets.")
    parser.add_argument(
        "--assets",
        default="param,wireless,lidar,rgb",
        help="Comma-separated asset names: param,wireless,lidar,rgb.",
    )
    parser.add_argument("--download-root", type=Path, default=Path("/root/datasets/DeepVerse/downloads"))
    parser.add_argument("--scenario-root", type=Path, default=Path("/root/datasets/DeepVerse/scenarios"))
    parser.add_argument("--scenario", default="DT31")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-mib", type=int, default=16)
    parser.add_argument("--verify-zip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--extract", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--generate-cache", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("dataset/deepverse_dt31/cache"))
    parser.add_argument(
        "--status",
        type=Path,
        default=Path("/root/datasets/DeepVerse/downloads/dt31_download_status.json"),
    )
    return parser.parse_args(argv)


def resolve_assets(value: str) -> list[Asset]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        raise SystemExit("--assets must name at least one asset.")
    unknown = [name for name in names if name not in ASSETS]
    if unknown:
        raise SystemExit(f"Unknown DT31 assets: {unknown}. Available: {sorted(ASSETS)}")
    return [ASSETS[name] for name in names]


def download_asset(
    *,
    asset: Asset,
    archive: Path,
    workers: int,
    chunk_bytes: int,
    status_path: Path,
) -> None:
    if archive.exists() and archive.stat().st_size == asset.size:
        log(f"{asset.filename} already complete ({format_bytes(asset.size)}); skipping download.")
        return
    if archive.exists():
        log(f"{asset.filename} has unexpected size {archive.stat().st_size}; replacing it.")
        archive.unlink()

    if not asset.ranged:
        download_full(asset, archive)
        return

    ranges = build_ranges(asset.size, chunk_bytes)
    parts_dir = archive.with_suffix(archive.suffix + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    log(
        f"Downloading {asset.filename}: {format_bytes(asset.size)} in {len(ranges)} parts "
        f"with {workers} workers."
    )
    started = time.time()
    completed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_part, asset, parts_dir, index, start, end): (index, start, end)
            for index, start, end in ranges
        }
        for future in concurrent.futures.as_completed(futures):
            index, size, status = future.result()
            completed += 1
            done_bytes = sum_completed_bytes(parts_dir, ranges)
            rate = done_bytes / max(time.time() - started, 1.0)
            payload = {
                "state": "downloading",
                "asset": asset.name,
                "completed_parts": completed,
                "total_parts": len(ranges),
                "downloaded_bytes": done_bytes,
                "total_bytes": asset.size,
                "mib_per_sec": rate / (1024 * 1024),
            }
            write_status(status_path, payload)
            log(
                f"{asset.filename}: {completed}/{len(ranges)} parts, part={index} {status}, "
                f"{format_bytes(done_bytes)}/{format_bytes(asset.size)}, "
                f"{rate / (1024 * 1024):.2f} MiB/s."
            )

    merge_parts(asset=asset, archive=archive, parts_dir=parts_dir, ranges=ranges)


def download_full(asset: Asset, archive: Path) -> None:
    log(f"Downloading {asset.filename} as a single file.")
    tmp = archive.with_suffix(archive.suffix + ".partial")
    if tmp.exists() and tmp.stat().st_size >= asset.size:
        tmp.unlink()
    cmd = [
        "curl",
        "--fail",
        "--location",
        "--http1.1",
        "--retry",
        "100",
        "--retry-delay",
        "5",
        "--retry-all-errors",
        "--connect-timeout",
        "30",
        "--speed-limit",
        "1024",
        "--speed-time",
        "60",
        "--output",
        str(tmp),
        asset.url,
    ]
    subprocess.run(cmd, check=True)
    if tmp.stat().st_size != asset.size:
        raise RuntimeError(f"{asset.filename} size {tmp.stat().st_size} != expected {asset.size}")
    tmp.replace(archive)
    log(f"{asset.filename} complete.")


def fetch_part(asset: Asset, parts_dir: Path, index: int, start: int, end: int) -> tuple[int, int, str]:
    part = parts_dir / f"part-{index:05d}.bin"
    expected = end - start + 1
    if part.exists() and part.stat().st_size == expected:
        return index, expected, "cached"
    if part.exists():
        part.unlink()

    for attempt in range(1, 31):
        if part.exists():
            part.unlink()
        cmd = [
            "curl",
            "--fail",
            "--location",
            "--http1.1",
            "--silent",
            "--show-error",
            "--retry",
            "5",
            "--retry-delay",
            "3",
            "--retry-all-errors",
            "--connect-timeout",
            "30",
            "--speed-limit",
            "1024",
            "--speed-time",
            "60",
            "--max-time",
            "600",
            "--range",
            f"{start}-{end}",
            "--output",
            str(part),
            asset.url,
        ]
        result = subprocess.run(cmd)
        if result.returncode == 0 and part.exists() and part.stat().st_size == expected:
            return index, expected, f"downloaded:{attempt}"

    got = part.stat().st_size if part.exists() else 0
    raise RuntimeError(f"Failed to download {asset.filename} part {index}: got {got}, expected {expected}.")


def build_ranges(size: int, chunk_bytes: int) -> list[tuple[int, int, int]]:
    if chunk_bytes <= 0:
        raise ValueError("--chunk-mib must be positive.")
    ranges: list[tuple[int, int, int]] = []
    start = 0
    index = 0
    while start < size:
        end = min(start + chunk_bytes - 1, size - 1)
        ranges.append((index, start, end))
        start = end + 1
        index += 1
    return ranges


def sum_completed_bytes(parts_dir: Path, ranges: list[tuple[int, int, int]]) -> int:
    total = 0
    for index, _, _ in ranges:
        part = parts_dir / f"part-{index:05d}.bin"
        if part.exists():
            total += part.stat().st_size
    return total


def merge_parts(*, asset: Asset, archive: Path, parts_dir: Path, ranges: list[tuple[int, int, int]]) -> None:
    tmp = archive.with_suffix(archive.suffix + ".merge")
    if tmp.exists():
        tmp.unlink()
    log(f"Merging {asset.filename}.")
    with tmp.open("wb") as dst:
        for index, start, end in ranges:
            part = parts_dir / f"part-{index:05d}.bin"
            expected = end - start + 1
            if not part.exists() or part.stat().st_size != expected:
                got = part.stat().st_size if part.exists() else 0
                raise RuntimeError(f"Cannot merge {asset.filename}: part {index} has {got}, expected {expected}.")
            with part.open("rb") as src:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
    if tmp.stat().st_size != asset.size:
        raise RuntimeError(f"Merged {asset.filename} size {tmp.stat().st_size} != expected {asset.size}.")
    tmp.replace(archive)
    log(f"{asset.filename} complete.")


def verify_zip(archive: Path) -> None:
    log(f"Verifying {archive.name}.")
    subprocess.run(["unzip", "-tq", str(archive)], check=True)
    log(f"{archive.name} verified.")


def extract_asset(*, asset: Asset, archive: Path, scenario_dir: Path, force: bool) -> None:
    marker_dir = scenario_dir / ".download_markers"
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / f"{asset.name}.extracted.json"
    target_dir = scenario_dir / asset.extract_subdir if asset.extract_subdir else scenario_dir
    if marker.exists() and not force:
        log(f"{asset.filename} already extracted according to {marker}; skipping extraction.")
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    log(f"Extracting {asset.filename} to {target_dir}.")
    extract_zip_safely(archive, target_dir)
    write_json(
        marker,
        {
            "asset": asset.name,
            "archive": str(archive),
            "archive_size": archive.stat().st_size,
            "target_dir": str(target_dir),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    log(f"{asset.filename} extracted.")


def extract_zip_safely(archive: Path, target_dir: Path) -> None:
    target_root = target_dir.resolve()
    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        for index, member in enumerate(members, start=1):
            name = member.filename
            if name in {"", "/", "\\"}:
                continue
            destination = (target_root / name).resolve()
            if not str(destination).startswith(str(target_root) + os.sep) and destination != target_root:
                raise RuntimeError(f"Unsafe path in {archive}: {name}")
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
            if index % 1000 == 0:
                log(f"Extracted {index}/{len(members)} entries from {archive.name}.")


def ensure_dt31_layout(scenario_dir: Path) -> None:
    """Normalize the public DT31 asset layout for the installed DeepVerse loader."""
    param_params = scenario_dir / "param" / "params.mat"
    wireless_params = scenario_dir / "wireless" / "params.mat"
    if wireless_params.exists():
        return
    if wireless_params.is_symlink():
        wireless_params.unlink()
    if not param_params.exists():
        return

    wireless_params.parent.mkdir(parents=True, exist_ok=True)
    try:
        wireless_params.symlink_to(Path("..") / "param" / "params.mat")
        log(f"Linked {wireless_params} -> ../param/params.mat.")
    except OSError:
        shutil.copy2(param_params, wireless_params)
        log(f"Copied {param_params} to {wireless_params}.")


def generate_cache(*, args: argparse.Namespace, scenario_dir: Path) -> None:
    config_m = scenario_dir / "param" / "config.m"
    if not config_m.exists():
        raise RuntimeError(f"Cannot generate DT31 cache; config.m is missing: {config_m}")
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    cmd = [
        "conda",
        "run",
        "-n",
        "kd_mm_beam",
        "python",
        str(ROOT / "scripts/deepverse/generate_dt31_cache.py"),
        "--config",
        str(ROOT / "configs/deepverse/dt31_generation.yaml"),
        "--scenario-root",
        str(args.scenario_root.expanduser().resolve()),
        "--scenario",
        args.scenario,
        "--config-m",
        str(config_m),
        "--output-root",
        str(output_root),
    ]
    log("Running DT31 cache generation.")
    subprocess.run(cmd, cwd=ROOT, check=True)
    log("DT31 cache generation complete.")


def write_status(path: Path, payload: dict[str, Any]) -> None:
    current: dict[str, Any] = {}
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current = {}
    current.update(payload)
    current["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(path, current)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def format_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{value} B"


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
