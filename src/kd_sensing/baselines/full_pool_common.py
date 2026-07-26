"""Shared helpers for the local Full-pool experiment workflows.

These utilities were previously copied into every `tools/run_full_pool_*.py`
entry point and into several `baselines` modules.  They are consolidated here so
that the local experiment surface has one implementation to reason about.

Serialization behaviour is deliberately *parameterized* rather than unified.
Existing run artifacts are SHA256-hashed and compared across rounds, so the JSON
key ordering and CSV strictness a caller used when it produced its manifests
must stay reproducible byte-for-byte:

- ``write_json`` defaults to insertion order (``sort_keys=False``), matching the
  BT-SCL/Candidate12/BTMA artifacts.  ``prototype_decision_adapter`` passes
  ``sort_keys=True`` to keep its own historical output.
- ``atomic_csv`` defaults to strict ``DictWriter`` behaviour.  Candidate12 passes
  ``extrasaction="ignore"`` because its row dicts intentionally carry extra keys.

This module owns no protocol constants: those live with the protocol itself in
``kd_sensing.data.mmw.full_pool_protocol``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


def now() -> str:
    """Return the current UTC timestamp used in every run manifest."""
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    """Seed every RNG the Full-pool workflows draw from, deterministically."""
    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256_json(value: Any) -> str:
    """Hash a JSON-serializable value with the canonical compact encoding."""
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Stream a file through SHA256 without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any, *, sort_keys: bool = False) -> None:
    """Write an indented JSON artifact, creating parent directories as needed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=True, indent=2, sort_keys=sort_keys)
    target.write_text(payload + "\n", encoding="utf-8")


def atomic_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
    *,
    extrasaction: str = "raise",
) -> None:
    """Write a CSV via a temporary file so readers never observe a partial run."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction=extrasaction)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, target)


__all__ = [
    "atomic_csv",
    "now",
    "set_seed",
    "sha256_file",
    "sha256_json",
    "write_json",
]
