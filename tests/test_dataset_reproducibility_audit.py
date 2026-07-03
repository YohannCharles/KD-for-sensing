import json
from pathlib import Path

from kd_sensing.diagnostics.dataset_reproducibility_audit import (
    audit_split_leakage,
    dataset_layout_descriptor,
    run_dataset_audit,
    write_audit_report,
)


def test_dataset_audit_reports_fields_files_labels_and_split_leakage(tmp_path: Path):
    root = tmp_path / "scenario31"
    (root / "camera").mkdir(parents=True)
    (root / "camera" / "0001.jpg").write_text("x", encoding="utf-8")
    csv_path = root / "train.csv"
    csv_path.write_text(
        "scene_id,sample_id,sequence_id,timestamp,split,camera1,gps1,beam1\n"
        "31,A,S1,1,train,camera/0001.jpg,1.0,1\n"
        "31,A,S1,2,test,camera/missing.jpg,1.1,64\n",
        encoding="utf-8",
    )

    report = run_dataset_audit(
        dataset_family="deepsense6g",
        data_root=root,
        csv_path=csv_path,
        scene=31,
        beam_shift=1,
    )

    assert report["read_only"] is True
    assert report["field_summary"]["camera_columns"] == ["camera1"]
    assert report["field_summary"]["sample_columns"] == ["sample_id"]
    assert report["file_reference_summary"]["camera"]["missing_count"] == 1
    assert report["label_summary"]["invalid_count"] == 0
    assert report["label_summary"]["label_space"] == "1-based-like"
    assert report["split_summary"]["status"] == "blocked"
    assert report["official_reproduction"]["status"] == "blocked"
    assert report["local_substitute"]["status"] == "ready"


def test_split_metadata_overlap_is_reported(tmp_path: Path):
    metadata = tmp_path / "split_metadata.json"
    metadata.write_text(
        json.dumps({"splits": {"train": {"sample_ids": ["a", "b"]}, "test": {"sample_ids": ["b", "c"]}}}),
        encoding="utf-8",
    )

    summary = audit_split_leakage([], split_metadata=metadata)

    assert summary["status"] == "blocked"
    assert summary["overlaps"][0]["examples"] == ["b"]


def test_mmw_layout_descriptor_records_required_subdirs(tmp_path: Path):
    root = tmp_path / "MMW" / "sunny"
    (root / "Sensor_Data").mkdir(parents=True)

    descriptor = dataset_layout_descriptor("mmw", data_root=root, condition="sunny")

    assert descriptor["dataset_family"] == "MMW"
    assert descriptor["required_subdirectory_status"]["Sensor_Data"] is True
    assert descriptor["required_subdirectory_status"]["Channel_Data"] is False
    assert descriptor["required_subdirectory_status"]["Prepared"] is False


def test_audit_report_writes_json_and_markdown(tmp_path: Path):
    report = run_dataset_audit(dataset_family="beambench", data_root=tmp_path / "missing", csv_path=None)

    outputs = write_audit_report(report, tmp_path / "out")

    assert Path(outputs["json"]).exists()
    assert Path(outputs["markdown"]).read_text(encoding="utf-8").startswith("# Dataset Reproducibility Audit")
