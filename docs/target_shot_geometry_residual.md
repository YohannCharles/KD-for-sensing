# Target-Shot Geometry-Residual Foundations

This change adds the data-contract foundations for auditable target-shot adaptation:

- deterministic source/target domain split artifacts
- 5% `target_labeled` selection plus `target_unlabeled` and `target_test` isolation
- geometry coarse beam and circular residual label utilities
- offline label distribution diagnostics

It does not add a residual neural network, residual predictor training loop, feature cache mainline, target prior calibration, or weather-aware fusion.

This foundation used to be demonstrated with `configs/hist_beam/*` smoke configs. Those configs are now retired; use the package CLIs with a current, explicit target-shot/GPS-anchor config when that workflow is reintroduced.

Minimal runtime-contract example:

```bash
conda run -n kd_mm_beam python - <<'PY'
from kd_sensing.data.dataset_runtime import RuntimeDataset, SampleIndex, SampleRow
from kd_sensing.data.geometry_residual import GeometryResidualTargetProvider

rows = [SampleRow(
    sample_id="demo",
    split="target_labeled",
    dataset_type="mmw",
    family="MMW",
    target_ref={"beam_abs": 3},
    metadata={"relative_geometry": {"available": True, "relative_azimuth": 90.0}},
)]
dataset = RuntimeDataset(
    sample_index=SampleIndex.from_rows(rows, storage_kind="demo"),
    modality_adapters=(),
    target_provider=GeometryResidualTargetProvider(num_beams=64, max_residual=8, num_geo_sectors=8),
    dataset_type="mmw",
    descriptor={"family": "MMW", "storage_kind": "demo"},
    enabled_modalities=(),
    input_profiles={},
)
print(dataset[0])
PY
```
