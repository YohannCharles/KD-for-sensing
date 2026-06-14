# Difficulty profiles

Difficulty profiles are current diagnostic/training profiles, not new modalities and not standalone result claims. Protocol status is tracked in `docs/experiment_protocols.md`; any claim using a profile must be recorded in `docs/result_claims_registry.md`.

| profile | status | boundary |
| --- | --- | --- |
| `clean_baseline.yaml` | diagnostic/training profile | clean comparison profile |
| `gps_mild_async_training.yaml` | training profile | perturbs GPS timing during training |
| `gps_severe_async_evaluation.yaml` | evaluation-only profile | severe async stress test, not a training result by itself |
| `gps_image_dropout_training.yaml` | training profile | controlled dropout robustness profile |
| `image_hard_degradation_sweep.yaml` | evaluation-only sweep | image degradation diagnostic sweep |

Profiles may perturb input tensors and reliability metadata such as `gps_valid_mask`, `gps_source_index` or image degradation metadata. They must not move `target_beam`, soft targets, sample id or split metadata. Outputs remain under ignored `outputs/`, `logs/` or benchmark output roots.
