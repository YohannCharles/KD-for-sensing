import csv
import json
from pathlib import Path

from kd_sensing.diagnostics.paper_artifact_export import (
    CLAIM_REGISTRY_COLUMNS,
    export_paper_artifacts,
    load_input_rows,
)


def test_paper_export_filters_main_rows_and_writes_manifest(tmp_path: Path):
    claims = tmp_path / "claims.csv"
    claims.write_text(
        "claim_id,method,dataset_split,target_source,metric,value,claim_status,seed_count,baseline,statistics,comparability,stress_status,provenance,caveat\n"
        "C1,JEPA,split-a,beam,DBA,0.88,local strict-validation,3,baseline-a,mean=0.88/std=0.01,comparable,passed,run-a,reviewed\n"
        "C2,Upper,split-a,beam,DBA,0.99,upper-bound,1,baseline-a,single seed,not_comparable,not_run,run-b,test selected\n"
        "C3,Smoke,split-a,beam,DBA,,mock/smoke,1,baseline-a,,not_comparable,smoke,run-c,synthetic\n",
        encoding="utf-8",
    )

    manifest = export_paper_artifacts([claims], output_dir=tmp_path / "paper")

    main_csv = Path(manifest["outputs"]["main_csv"])
    appendix_csv = Path(manifest["outputs"]["appendix_csv"])
    main_rows = list(csv.DictReader(main_csv.open(encoding="utf-8")))
    appendix_rows = list(csv.DictReader(appendix_csv.open(encoding="utf-8")))

    assert [row["claim_id"] for row in main_rows] == ["C1"]
    assert {row["claim_id"] for row in appendix_rows} == {"C2", "C3"}
    assert Path(manifest["outputs"]["main_markdown"]).exists()
    assert Path(manifest["outputs"]["main_latex"]).exists()
    assert Path(manifest["outputs"]["excluded_report_csv"]).exists()
    assert Path(manifest["manifest_path"]).exists()


def test_paper_export_hard_excludes_unverified_and_candidate_only_rows(tmp_path: Path):
    claims = tmp_path / "claims.csv"
    claims.write_text(
        "claim_id,method,dataset_split,target_source,metric,value,claim_status,candidate_only,seed_count,baseline,statistics,comparability,stress_status,provenance,caveat\n"
        "C1,Reviewed,split-a,beam,DBA,0.88,local strict-validation,false,3,base,mean/std,comparable,passed,run-a,reviewed\n"
        "C2,Unverified,split-a,beam,DBA,0.70,unverified,false,3,base,mean/std,pending,pending,run-b,needs audit\n"
        "C3,Draft,split-a,beam,DBA,0.71,local strict-validation,true,3,base,mean/std,comparable,passed,run-c,draft only\n",
        encoding="utf-8",
    )

    manifest = export_paper_artifacts(
        [claims],
        output_dir=tmp_path / "paper",
        include_statuses=["unverified"],
    )

    main_rows = list(csv.DictReader(Path(manifest["outputs"]["main_csv"]).open(encoding="utf-8")))
    excluded_rows = list(csv.DictReader(Path(manifest["outputs"]["excluded_report_csv"]).open(encoding="utf-8")))

    assert [row["claim_id"] for row in main_rows] == ["C1"]
    assert {row["claim_id"] for row in excluded_rows} == {"C2", "C3"}
    assert any("status_not_reviewed" in row["exclusion_reason"] for row in excluded_rows if row["claim_id"] == "C2")
    assert any("candidate_only=true" in row["exclusion_reason"] for row in excluded_rows if row["claim_id"] == "C3")
    assert Path(manifest["outputs"]["diagnostic_rows_csv"]).exists()


def test_paper_export_writes_stress_and_pattern_figure_data(tmp_path: Path):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "claim_id": "S1",
                        "method": "predictive",
                        "condition": "image_noise",
                        "severity": "0.5",
                        "metric": "DBA",
                        "mean": "0.7",
                        "std": "0.02",
                        "claim_status": "local experimental baseline",
                    },
                    {
                        "claim_id": "P1",
                        "method": "rbma",
                        "pattern": "missing_gps",
                        "available_mask": "image,radar,lidar",
                        "metric": "top1",
                        "value": "0.4",
                        "sample_count": "12",
                        "claim_status": "local experimental baseline",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    manifest = export_paper_artifacts([summary], output_dir=tmp_path / "paper")

    assert Path(manifest["outputs"]["stress_curve_csv"]).exists()
    assert Path(manifest["outputs"]["stress_curve_json"]).exists()
    assert Path(manifest["outputs"]["pattern_heatmap_csv"]).exists()
    assert Path(manifest["outputs"]["pattern_heatmap_json"]).exists()


def test_markdown_claim_registry_rows_are_loaded(tmp_path: Path):
    registry = tmp_path / "registry.md"
    registry.write_text(
        "| claim_id | model line | dataset / split | target / metric field | value summary | claim status | caveat |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| `C1` | Model | Split | DBA | 0.5 | `local substitute` | caveat |\n",
        encoding="utf-8",
    )

    rows = load_input_rows(registry)

    assert rows[0]["claim_id"] == "C1"
    assert rows[0]["claim status"] == "local substitute"


def test_paper_export_rejects_unknown_empty_and_incomplete_reviewed_statuses(tmp_path: Path):
    claims = tmp_path / "claims.csv"
    claims.write_text(
        "claim_id,method,dataset_split,target_source,metric,value,claim_status,seed_count,baseline,statistics,comparability,stress_status,provenance,caveat\n"
        "EMPTY,M,split,beam,DBA,0.1,,3,B,mean/std,comparable,passed,run,caveat\n"
        "UNKNOWN,M,split,beam,DBA,0.1,new-status,3,B,mean/std,comparable,passed,run,caveat\n"
        "INCOMPLETE,M,split,beam,DBA,0.1,local strict-validation,3,B,,comparable,passed,run,caveat\n",
        encoding="utf-8",
    )

    manifest = export_paper_artifacts([claims], output_dir=tmp_path / "paper")
    main = list(csv.DictReader(Path(manifest["outputs"]["main_csv"]).open(encoding="utf-8")))
    excluded = list(csv.DictReader(Path(manifest["outputs"]["excluded_report_csv"]).open(encoding="utf-8")))

    assert main == []
    reasons = {row["claim_id"]: row["exclusion_reason"] for row in excluded}
    assert reasons["EMPTY"] == "status_not_reviewed:empty"
    assert reasons["UNKNOWN"] == "status_not_reviewed:new-status"
    assert "missing_required_fields:statistics" in reasons["INCOMPLETE"]


def test_real_claim_registry_schema_and_catalog_foreign_keys_are_complete():
    root = Path(__file__).resolve().parents[1]
    registry_rows = load_input_rows(root / "docs" / "result_claims_registry.md")
    catalog_rows = load_input_rows(root / "docs" / "mainline_model_catalog.md")

    assert registry_rows
    assert tuple(registry_rows[0]) == CLAIM_REGISTRY_COLUMNS
    claim_ids = [row["claim_id"] for row in registry_rows]
    assert len(claim_ids) == len(set(claim_ids))
    assert {row["claim"] for row in catalog_rows if row.get("claim")} <= set(claim_ids)
