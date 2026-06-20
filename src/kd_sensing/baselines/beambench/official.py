import json
import subprocess
import sys
from pathlib import Path
from typing import Any


OFFICIAL_REPOSITORY_URL = "https://github.com/ITU-AI-ML-in-5G-Challenge/BeamBench"
OFFICIAL_COMMIT = "8e2c29a2afc898a69b9f9f7ece039d1e48ba60e8"
README_EVAL_COMMAND = "python3 challenge.py --gpu_id 0 --data_folder ./raw_data/test/ --csv ml_challenge_test_multi_modal.csv"
DEFAULT_DATA_FOLDER = "./raw_data/test/"
DEFAULT_MODEL_DIR = "results/models"
DEFAULT_PREDICTION_DIR = "results/topk"

EXPECTED_SOURCE_FILES = (
    "README.md",
    "Dockerfile",
    "challenge.py",
    "challenge_lstm.py",
    "classical.py",
    "config/camera_ae.cfg",
    "config/gps_dense.cfg",
    "libraries/general.py",
    "models/ae_camera_model.py",
    "models/dense_model.py",
)
REFERENCED_BUT_MISSING_SOURCE_FILES = (
    "models/ae_lidar_model.py",
    "models/ae_radar_model.py",
    "models/cl_camera_model.py",
    "models/cl_radar_model.py",
    "models/lstm_model.py",
    "models/mmWave_camera_model.py",
    "models/mmWave_lidar_model.py",
    "models/mmWave_radar_model.py",
    "config/camera_cl.cfg",
    "config/radar_ae.cfg",
    "config/radar_cl.cfg",
    "config/lidar_ae.cfg",
    "config/lstm_model.cfg",
    "config/camera_mmWave.cfg",
    "config/radar_mmWave.cfg",
    "config/lidar_mmWave.cfg",
)


def audit_official_repository(root: str | Path) -> dict[str, Any]:
    base = Path(root)
    commit = _git_commit(base)
    expected = {path: (base / path).exists() for path in EXPECTED_SOURCE_FILES}
    missing_referenced = [path for path in REFERENCED_BUT_MISSING_SOURCE_FILES if not (base / path).exists()]
    pyc_only = sorted(
        path.relative_to(base).as_posix()
        for path in (base / "models" / "__pycache__").glob("*.pyc")
        if not (base / "models" / (path.name.split(".cpython-")[0] + ".py")).exists()
    )
    return {
        "official_repository_url": OFFICIAL_REPOSITORY_URL,
        "official_commit": commit or OFFICIAL_COMMIT,
        "expected_commit": OFFICIAL_COMMIT,
        "clone_path": str(base),
        "readme_eval_command": README_EVAL_COMMAND,
        "default_data_folder": DEFAULT_DATA_FOLDER,
        "default_model_dir": DEFAULT_MODEL_DIR,
        "default_prediction_dir": DEFAULT_PREDICTION_DIR,
        "expected_source_files": expected,
        "referenced_but_missing_source_files": missing_referenced,
        "pyc_only_model_artifacts": pyc_only,
        "official_environment": {
            "ubuntu": "18.04",
            "cuda": "11.4",
            "docker_base": "nvidia/cuda:11.4.2-runtime-ubuntu18.04",
            "python": "3.7",
            "pytorch_wheel": "torch/torchvision/torchaudio from cu113 index",
            "key_pip_dependencies": [
                "numpy",
                "Pillow",
                "matplotlib",
                "utm",
                "opencv-python",
                "sklearn",
                "tqdm",
                "pandas",
                "future",
                "open3d",
                "h5py",
            ],
        },
    }


def plan_official_evaluation(
    *,
    official_root: str | Path,
    data_folder: str | Path,
    csv: str = "ml_challenge_test_multi_modal.csv",
    type_list: str = "radar_dense_camera_ae_gps",
    seed: int = 42,
    adapt: str = "adapt_",
    gpu_id: int = 0,
    output_dir: str | Path | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    root = Path(official_root)
    data_root = Path(data_folder)
    command = [
        "python3",
        "challenge.py",
        "--gpu_id",
        str(int(gpu_id)),
        "--data_folder",
        str(data_root),
        "--csv",
        str(csv),
        "--type_list",
        str(type_list),
        "--seed",
        str(int(seed)),
        "--root",
        str(root),
        "--adapt",
        str(adapt),
    ]
    missing = []
    if not root.exists():
        missing.append(f"official_root:{root}")
    if not (root / "challenge.py").exists():
        missing.append("challenge.py")
    if not (data_root / csv).exists():
        missing.append(f"data_csv:{data_root / csv}")
    missing.extend(_missing_checkpoint_paths(root, type_list=type_list, seed=seed, adapt=adapt))
    audit = audit_official_repository(root) if root.exists() else {}
    missing.extend(audit.get("referenced_but_missing_source_files", []))
    prediction_path = root / DEFAULT_PREDICTION_DIR / f"fusion_{adapt}{type_list}_{seed}.csv"
    report: dict[str, Any] = {
        "mode": "official_evaluation",
        "official_commit": audit.get("official_commit", OFFICIAL_COMMIT),
        "command": command,
        "command_text": " ".join(command),
        "cwd": str(root),
        "data_folder": str(data_root),
        "csv": str(csv),
        "type_list": str(type_list),
        "seed": int(seed),
        "checkpoint_dir": str(root / DEFAULT_MODEL_DIR),
        "prediction_path": str(prediction_path),
        "execute": bool(execute),
        "blocked": bool(missing),
        "blocked_reasons": sorted(dict.fromkeys(str(item) for item in missing)),
        "audit": audit,
    }
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "official_eval_plan.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if execute and missing:
        report["returncode"] = 2
        return report
    if execute:
        result = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
        report["returncode"] = int(result.returncode)
        report["stdout_tail"] = result.stdout[-4000:]
        report["stderr_tail"] = result.stderr[-4000:]
    return report


def plan_official_classical_evaluation(
    *,
    official_root: str | Path,
    data_folder: str | Path,
    csv: str = "ml_challenge_test_multi_modal.csv",
    gpu_id: int = 0,
    beams_shift: int = 1,
    output_dir: str | Path | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    root = Path(official_root)
    data_root = Path(data_folder)
    command = [
        "python3",
        "classical.py",
        "--gpu_id",
        str(int(gpu_id)),
        "--data_folder",
        str(data_root),
        "--csv",
        str(csv),
        "--root",
        str(root),
        "--beams_shift",
        str(int(beams_shift)),
    ]
    missing = []
    if not root.exists():
        missing.append(f"official_root:{root}")
    if not (root / "classical.py").exists():
        missing.append("classical.py")
    if not (data_root / csv).exists():
        missing.append(f"data_csv:{data_root / csv}")
    for filename in ("classic.npy", "classic_angle.npy", "classic_corr.npy"):
        path = root / DEFAULT_MODEL_DIR / filename
        if not path.exists():
            missing.append(f"checkpoint:{path}")
    audit = audit_official_repository(root) if root.exists() else {}
    prediction_path = root / DEFAULT_PREDICTION_DIR / "classical.csv"
    report: dict[str, Any] = {
        "mode": "official_classical_evaluation",
        "official_commit": audit.get("official_commit", OFFICIAL_COMMIT),
        "command": command,
        "command_text": " ".join(command),
        "cwd": str(root),
        "data_folder": str(data_root),
        "csv": str(csv),
        "beams_shift": int(beams_shift),
        "checkpoint_dir": str(root / DEFAULT_MODEL_DIR),
        "prediction_path": str(prediction_path),
        "execute": bool(execute),
        "blocked": bool(missing),
        "blocked_reasons": sorted(dict.fromkeys(str(item) for item in missing)),
        "audit": audit,
    }
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "official_classical_plan.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if execute and missing:
        report["returncode"] = 2
        return report
    if execute:
        result = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
        report["returncode"] = int(result.returncode)
        report["stdout_tail"] = result.stdout[-4000:]
        report["stderr_tail"] = result.stderr[-4000:]
    return report


def _missing_checkpoint_paths(root: Path, *, type_list: str, seed: int, adapt: str) -> list[str]:
    model_dir = root / DEFAULT_MODEL_DIR
    missing = []
    direct_tokens = ("gps_dense", "camera_mmWave", "radar_mmWave", "lidar_mmWave")
    direct = any(token in type_list for token in direct_tokens)
    token_to_checkpoint = {
        "camera_ae": f"adapt_bb_camera_ae_0_{seed}.pth",
        "camera_cl": f"adapt_bb_camera_cl_0_{seed}.pth",
        "radar_ae": f"adapt_bb_radar_ae_0_{seed}.pth",
        "radar_cl": f"adapt_bb_radar_cl_0_{seed}.pth",
        "lidar_ae": f"adapt_bb_lidar_ae_0_{seed}.pth",
        "gps_dense": f"{adapt}gps_dense_0_{seed}.pth",
        "camera_mmWave": f"{adapt}camera_mmWave_0_{seed}.pth",
        "radar_mmWave": f"{adapt}radar_mmWave_0_{seed}.pth",
        "lidar_mmWave": f"{adapt}lidar_mmWave_0_{seed}.pth",
    }
    for token, filename in token_to_checkpoint.items():
        if token in type_list and not (model_dir / filename).exists():
            missing.append(f"checkpoint:{model_dir / filename}")
    if not direct:
        fusion = model_dir / f"fusion_{adapt}{type_list}_0_{seed}.pth"
        if not fusion.exists():
            missing.append(f"checkpoint:{fusion}")
    return missing


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None
