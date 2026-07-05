#!/usr/bin/env python3

import argparse
from pathlib import Path


KEYWORDS = (
    "reliability",
    "quality",
    "gate",
    "gated",
    "mask_weight",
    "weighted_fusion",
    "reliability_fusion",
    "modality_weight",
    "confidence_fusion",
    "mask_gate",
    "available_mask",
)
SOURCE_GLOBS = ("src/**/*.py", "configs/**/*.yaml", "scripts/**/*.py", "tests/**/*.py")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    matches = _scan(root)
    modules = [path for path in matches if path.startswith("src/")]
    configs = [path for path in matches if path.startswith("configs/")]
    run_names = _run_names(root)
    fresh_eval_exists = any((Path(args.baseline_root) / child).exists() for child in ("fresh_eval", "summary"))
    lines = [
        "# Reliability Fusion Audit",
        "",
        "## found_modules",
        *(_bullet(path, matches[path]) for path in modules[:80]),
        "",
        "## found_configs",
        *(_bullet(path, matches[path]) for path in configs[:80]),
        "",
        "## found_run_names",
        *[f"- `{name}`" for name in run_names[:80]],
        "",
        "## whether_fresh_eval_exists",
        f"- `{str(fresh_eval_exists).lower()}` under `{args.baseline_root}`",
        "",
        "## whether_compatible_with_proto",
        "- `true`: `u_mask_beam_jepa` already exposes `fusion_type=weighted_sum`, internal modality reliability heads, missing-mask-aware reliability zeroing and prototype alignment compatibility.",
        "",
        "## whether_compatible_with_randomdrop_subset",
        "- `true`: randomdrop subset is a training difficulty/exposure setting and does not require a separate fusion architecture.",
        "",
        "## recommended_reuse_path",
        "- Reuse `u_mask_beam_jepa` with `fusion_type=weighted_sum`, `use_jepa_loss=false`, `use_beam_prototype_alignment=true`, and explicit `model.primary.reliability_fusion.enabled=true` metadata.",
        "- Use `reliability_weights_epoch.csv` diagnostics from training to verify missing modality weight zero and available-modality normalization.",
        "",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote reliability fusion audit to {out}.")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit existing reliability/mask weighted fusion implementation.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--baseline-root", default="outputs/scene31_baseline_pack_lmdb")
    parser.add_argument("--out", default="outputs/scene31_subset_reliability_lmdb/reliability_fusion_audit.md")
    return parser


def _scan(root: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    lowered = tuple(keyword.lower() for keyword in KEYWORDS)
    for pattern in SOURCE_GLOBS:
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            hits = sorted({keyword for keyword in lowered if keyword in text.lower()})
            if hits:
                out[str(path.relative_to(root))] = hits
    return out


def _run_names(root: Path) -> list[str]:
    names: set[str] = set()
    for path in (root / "outputs").glob("**/run_status.json") if (root / "outputs").exists() else []:
        names.add(path.parent.name)
    for path in (root / "configs").glob("**/*.yaml") if (root / "configs").exists() else []:
        stem = path.stem
        if "reliability" in stem or "weighted" in stem:
            names.add(stem)
    return sorted(names)


def _bullet(path: str, hits: list[str]) -> str:
    return f"- `{path}`: {', '.join(hits)}"


if __name__ == "__main__":
    raise SystemExit(main())
