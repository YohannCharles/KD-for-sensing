## remove-distillation-support migration notes

### Config path migration

| Old surface | New surface |
| --- | --- |
| `configs/<modality>/teacher_no_kd.yaml` | `configs/<modality>/strong.yaml` |
| `configs/<modality>/student_no_kd.yaml` | `configs/<modality>/lightweight.yaml` |
| `configs/<modality>/no_kd.yaml` | `configs/<modality>/supervised.yaml` |
| `configs/<modality>/logits_kd.yaml` | rejected; use `strong.yaml`, `lightweight.yaml`, or `supervised.yaml` |
| `configs/<modality>/rkd.yaml` | rejected; use `strong.yaml`, `lightweight.yaml`, or `supervised.yaml` |
| `configs/fusion/<slug>_teacher_no_kd.yaml` | `configs/fusion/<slug>_strong.yaml` |
| `configs/fusion/<slug>_student_no_kd.yaml` | `configs/fusion/<slug>_lightweight.yaml` |
| `configs/fusion/<slug>_no_kd.yaml` | rejected for canonical fusion; use `<slug>_strong.yaml` or `<slug>_lightweight.yaml` |
| `configs/fusion/<slug>_logits_kd.yaml` | rejected; use `<slug>_lightweight.yaml` or an adaptation workflow |
| `configs/fusion/<slug>_rkd.yaml` | rejected; use `<slug>_lightweight.yaml` or an adaptation workflow |
| `configs/<modality>/snapshot_next_frame_no_kd.yaml` | `configs/<modality>/snapshot_next_frame_supervised.yaml` |
| `configs/fusion/<slug>_snapshot_next_frame_no_kd.yaml` | `configs/fusion/<slug>_snapshot_next_frame_supervised.yaml` |
| `configs/fusion/<slug>_<objective>_no_kd.yaml` | `configs/fusion/<slug>_<objective>_supervised.yaml` |
| advanced or CSI paths containing `_no_kd` | replace `_no_kd` with `_supervised`, unless a more specific workflow name already exists |

### Model and registry migration

| Old registry/config name | New registry/config name |
| --- | --- |
| `image_teacher` | `image_strong` |
| `image_student` | `image_lightweight` |
| `radar_teacher` | `radar_strong` |
| `radar_student` | `radar_lightweight` |
| `gps_teacher` | `gps_strong` |
| `gps_student` | `gps_lightweight` |
| `lidar_teacher` | `lidar_strong` |
| `lidar_student` | `lidar_lightweight` |
| `mmwave_teacher` | `mmwave_strong` |
| `mmwave_student` | `mmwave_lightweight` |
| `fusion_teacher` | `fusion_strong` |
| `fusion_student` | `fusion_lightweight` |
| `DISTILLERS`, `no_kd`, `logits_kd`, `rkd`, `g2d` | removed; supervised/adaptation losses live under loss/objective/extension code |
| `model.teacher` / `model.student` | `model.primary` |

Internal class names may remain temporarily where renaming would create avoidable churn, but public configs, registry names, tests, and docs must use the new names.

### Historical artifact boundary

This change does not delete or rewrite `outputs/`, `logs/`, `All_models/`, `dataset/`, caches, checkpoints, or archived OpenSpec artifacts. Historical files may still contain old KD metadata for read-only inspection, but new training/evaluation outputs must not emit KD metadata fields.
