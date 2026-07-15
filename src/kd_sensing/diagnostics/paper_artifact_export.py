import csv
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("outputs/paper_artifacts")
MAIN_TABLE_COLUMNS = (
    "claim_id",
    "method",
    "dataset_split",
    "target_source",
    "metric",
    "value",
    "claim_status",
    "seed_count",
    "baseline",
    "statistics",
    "comparability",
    "stress_status",
    "provenance",
    "caveat",
)
REVIEWED_MAIN_STATUSES = (
    "official-reproduction",
    "local-strict-validation",
    "local-experimental",
)
REVIEWED_REQUIRED_FIELDS = (
    "claim_id",
    "method",
    "dataset_split",
    "target_source",
    "metric",
    "value",
    "seed_count",
    "baseline",
    "statistics",
    "comparability",
    "stress_status",
    "provenance",
    "caveat",
)
CLAIM_REGISTRY_COLUMNS = (
    "claim_id",
    "model_line",
    "dataset_split",
    "target_source",
    "metric",
    "value",
    "claim_status",
    "run_date_commit",
    "config_provenance",
    "checkpoint_provenance",
    "caveat",
    "seed_count",
    "baseline",
    "statistics",
    "comparability",
    "stress_status",
    "candidate_only",
    "upgrade_gate",
)
EXCLUDED_REPORT_COLUMNS = ("claim_id", "claim_status", "candidate_only", "caveat", "exclusion_reason")
STRESS_COLUMNS = ("condition", "severity", "method", "metric", "mean", "std", "ci", "claim_status", "caveat")
HEATMAP_COLUMNS = (
    "pattern",
    "available_mask",
    "method",
    "metric",
    "value",
    "sample_count",
    "metric_profile",
    "claim_status",
    "caveat",
)


def export_paper_artifacts(
    input_paths: list[str | Path],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    include_statuses: list[str] | tuple[str, ...] = (),
    table_name: str = "main_results",
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    table_dir = out_dir / "tables"
    figure_dir = out_dir / "figure_data"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for input_path in input_paths:
        path = Path(input_path)
        loaded = load_input_rows(path)
        if not loaded:
            warnings.append(f"no rows loaded from {path}")
        raw_rows.extend({**row, "_source_file": str(path)} for row in loaded)

    normalized = [normalize_claim_row(row) for row in raw_rows]
    main_rows, appendix_rows = filter_main_rows(normalized, include_statuses=include_statuses)

    outputs: dict[str, str] = {}
    outputs["main_csv"] = str(write_csv(table_dir / f"{table_name}.csv", main_rows, MAIN_TABLE_COLUMNS))
    outputs["main_markdown"] = str(write_markdown_table(table_dir / f"{table_name}.md", main_rows, MAIN_TABLE_COLUMNS))
    outputs["main_latex"] = str(write_latex_table(table_dir / f"{table_name}.tex", main_rows, MAIN_TABLE_COLUMNS))
    outputs["appendix_csv"] = str(write_csv(table_dir / "appendix_rows.csv", appendix_rows, MAIN_TABLE_COLUMNS))
    outputs["appendix_markdown"] = str(write_markdown_table(table_dir / "appendix_rows.md", appendix_rows, MAIN_TABLE_COLUMNS))
    excluded_report = excluded_report_rows(appendix_rows)
    outputs["excluded_report_csv"] = str(write_csv(table_dir / "excluded_report.csv", excluded_report, EXCLUDED_REPORT_COLUMNS))
    outputs["excluded_report_json"] = str(write_json(table_dir / "excluded_report.json", excluded_report))
    outputs["excluded_report_markdown"] = str(
        write_markdown_table(table_dir / "excluded_report.md", excluded_report, EXCLUDED_REPORT_COLUMNS)
    )
    diagnostic_rows = [row for row in appendix_rows if row.get("_explicit_appendix")]
    if diagnostic_rows:
        outputs["diagnostic_rows_csv"] = str(write_csv(table_dir / "diagnostic_rows.csv", diagnostic_rows, MAIN_TABLE_COLUMNS))
        outputs["diagnostic_rows_markdown"] = str(
            write_markdown_table(table_dir / "diagnostic_rows.md", diagnostic_rows, MAIN_TABLE_COLUMNS)
        )

    stress_rows = normalize_stress_rows(raw_rows)
    if stress_rows:
        outputs["stress_curve_csv"] = str(write_csv(figure_dir / "stress_curve.csv", stress_rows, STRESS_COLUMNS))
        outputs["stress_curve_json"] = str(write_json(figure_dir / "stress_curve.json", stress_rows))
    heatmap_rows = normalize_heatmap_rows(raw_rows)
    if heatmap_rows:
        outputs["pattern_heatmap_csv"] = str(write_csv(figure_dir / "pattern_heatmap.csv", heatmap_rows, HEATMAP_COLUMNS))
        outputs["pattern_heatmap_json"] = str(write_json(figure_dir / "pattern_heatmap.json", heatmap_rows))

    manifest = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_files": [str(Path(path)) for path in input_paths],
        "input_claim_ids": [row["claim_id"] for row in normalized if row.get("claim_id")],
        "filter": {
            "reviewed_main_statuses": list(REVIEWED_MAIN_STATUSES),
            "reviewed_required_fields": list(REVIEWED_REQUIRED_FIELDS),
            "include_statuses": list(include_statuses),
            "main_row_count": len(main_rows),
            "appendix_row_count": len(appendix_rows),
            "excluded_row_count": len(excluded_report),
            "diagnostic_row_count": len(diagnostic_rows),
        },
        "outputs": outputs,
        "warnings": warnings,
    }
    manifest_path = out_dir / "paper_export_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def load_input_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"paper export input not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _rows_from_json(json.loads(path.read_text(encoding="utf-8")))
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix in {".md", ".markdown", ".txt"}:
        return _rows_from_markdown_table(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported paper export input format: {path}")


def normalize_claim_row(row: dict[str, Any]) -> dict[str, str]:
    provenance_parts = [
        _string(_pick(row, "provenance", "config_provenance", "config / runner", "_source_file")),
        _string(_pick(row, "run_date_commit", "run date or commit", "commit")),
        _string(_pick(row, "checkpoint_provenance", "checkpoint provenance")),
    ]
    return {
        "claim_id": _string(_pick(row, "claim_id", "claim id", "id", "claim")),
        "method": _string(_pick(row, "method", "model line", "model_line", "model", "line", "method_name", "subject")),
        "dataset_split": _string(_pick(row, "dataset / split", "dataset_split", "dataset", "split")),
        "target_source": _string(_pick(row, "target_source", "target source", "label source")),
        "metric": _string(
            _pick(row, "metric", "target / metric field", "target_metric", "metric_field", "metric_profile")
        ),
        "value": _string(_pick(row, "value", "value summary", "result", "score", "mean")),
        "claim_status": _string(_pick(row, "claim_status", "claim status", "status")),
        "candidate_only": _string(_pick(row, "candidate_only", "candidate only")),
        "seed_count": _string(_pick(row, "seed_count", "seed count", "seeds")),
        "baseline": _string(_pick(row, "baseline", "comparison baseline")),
        "statistics": _string(_pick(row, "statistics", "mean/std", "mean_std", "ci", "confidence_interval")),
        "comparability": _string(_pick(row, "comparability", "comparability status")),
        "stress_status": _string(_pick(row, "stress_status", "stress status", "stress suite status")),
        "provenance": "; ".join(part for part in provenance_parts if part),
        "caveat": _string(_pick(row, "caveat", "note", "notes", "warning", "warnings")),
    }


def filter_main_rows(
    rows: list[dict[str, str]],
    *,
    include_statuses: list[str] | tuple[str, ...] = (),
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    include_markers = tuple(marker.lower() for marker in include_statuses)
    main_rows: list[dict[str, str]] = []
    appendix_rows: list[dict[str, str]] = []
    for row in rows:
        status = row.get("claim_status", "").lower()
        explicit_include = bool(include_markers and any(marker in status for marker in include_markers))
        exclusion_reason = _exclusion_reason(row)
        if exclusion_reason:
            appendix_rows.append({**row, "exclusion_reason": exclusion_reason, "_explicit_appendix": explicit_include})
        else:
            main_rows.append(row)
    return main_rows, appendix_rows

def excluded_report_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "claim_id": row.get("claim_id", ""),
            "claim_status": row.get("claim_status", ""),
            "candidate_only": row.get("candidate_only", ""),
            "caveat": row.get("caveat", ""),
            "exclusion_reason": row.get("exclusion_reason", ""),
        }
        for row in rows
    ]

def _exclusion_reason(row: dict[str, str]) -> str:
    reasons: list[str] = []
    status = _canonical_status(row.get("claim_status", ""))
    if _truthy(row.get("candidate_only")):
        reasons.append("candidate_only=true")
    if status not in REVIEWED_MAIN_STATUSES:
        reasons.append(f"status_not_reviewed:{status or 'empty'}")
    else:
        missing = [field for field in REVIEWED_REQUIRED_FIELDS if not str(row.get(field, "")).strip()]
        if missing:
            reasons.append("missing_required_fields:" + ",".join(missing))
    return "; ".join(dict.fromkeys(reasons))


def _canonical_status(value: Any) -> str:
    return "-".join(str(value or "").strip().lower().replace("_", "-").split())


def normalize_stress_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        if _pick(row, "condition", "stress_condition") is None:
            continue
        if _pick(row, "severity", "level", "stress_level") is None:
            continue
        out.append(
            {
                "condition": _string(_pick(row, "condition", "stress_condition")),
                "severity": _string(_pick(row, "severity", "level", "stress_level")),
                "method": _string(_pick(row, "method", "model", "model line")),
                "metric": _string(_pick(row, "metric", "metric_field", "target / metric field")),
                "mean": _string(_pick(row, "mean", "value", "score")),
                "std": _string(_pick(row, "std", "stderr", "standard_deviation")),
                "ci": _string(_pick(row, "ci", "confidence_interval")),
                "claim_status": _string(_pick(row, "claim_status", "claim status", "status")),
                "caveat": _string(_pick(row, "caveat", "note", "notes")),
            }
        )
    return out


def normalize_heatmap_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        if _pick(row, "pattern", "missing_pattern") is None:
            continue
        out.append(
            {
                "pattern": _string(_pick(row, "pattern", "missing_pattern")),
                "available_mask": _string(_pick(row, "available_mask", "mask", "modality_mask")),
                "method": _string(_pick(row, "method", "model", "model line")),
                "metric": _string(_pick(row, "metric", "metric_field", "target / metric field")),
                "value": _string(_pick(row, "value", "mean", "score")),
                "sample_count": _string(_pick(row, "sample_count", "count", "num_samples")),
                "metric_profile": _string(_pick(row, "metric_profile", "metric profile")),
                "claim_status": _string(_pick(row, "claim_status", "claim status", "status")),
                "caveat": _string(_pick(row, "caveat", "note", "notes")),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, str]], columns: tuple[str, ...]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return path


def write_markdown_table(path: Path, rows: list[dict[str, str]], columns: tuple[str, ...]) -> Path:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_cell(row.get(column, "")) for column in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_latex_table(path: Path, rows: list[dict[str, str]], columns: tuple[str, ...]) -> Path:
    align = "l" * len(columns)
    lines = [f"\\begin{{tabular}}{{{align}}}", r"\hline", " & ".join(columns) + r" \\", r"\hline"]
    for row in rows:
        lines.append(" & ".join(_latex_cell(row.get(column, "")) for column in columns) + r" \\")
    lines.extend([r"\hline", "\\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _rows_from_json(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("claims", "rows", "summary", "metrics", "conditions"):
        value = payload.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, dict)]
    return [payload]


def _rows_from_markdown_table(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    header: list[str] | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        cells = [_strip_markdown_cell(cell) for cell in stripped.strip("|").split("|")]
        if not cells:
            continue
        if all(set(cell) <= {"-", ":", " "} for cell in cells):
            continue
        if header is None:
            header = cells
            continue
        if len(cells) == len(header):
            rows.append(dict(zip(header, cells)))
    return rows


def _pick(row: dict[str, Any], *names: str) -> Any:
    lowered = {str(key).strip().lower(): value for key, value in row.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value not in (None, ""):
            return value
    return None


def _string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value).strip()


def _strip_markdown_cell(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == "`" and text[-1] == "`":
        return text[1:-1]
    return text.replace(r"\|", "|")


def _markdown_cell(value: str) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _latex_cell(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    text = str(value).replace("\n", " ")
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:
        return "unavailable"
    return result.stdout.strip() or "unavailable"
