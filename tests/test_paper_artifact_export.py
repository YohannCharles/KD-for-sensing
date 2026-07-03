import csv
import json
from pathlib import Path

from kd_sensing.diagnostics.paper_artifact_export import export_paper_artifacts, load_input_rows


def test_paper_export_filters_main_rows_and_writes_manifest(tmp_path: Path):
    claims = tmp_path / "claims.csv"
    claims.write_text(
        "claim_id,method,dataset_split,metric,value,claim_status,provenance,caveat\n"
        "C1,JEPA,split-a,DBA,0.88,local strict-validation,run-a,reviewed\n"
        "C2,Upper,split-a,DBA,0.99,upper-bound,run-b,test selected\n"
        "C3,Smoke,split-a,DBA,,mock/smoke,run-c,synthetic\n",
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
    assert Path(manifest["manifest_path"]).exists()


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
