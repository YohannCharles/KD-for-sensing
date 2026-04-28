# Extension Guide

KD sensing uses lightweight registries. A config selects a component by `type`, and the remaining fields are passed to that component constructor.

## Inspect Available Components

```python
from kd_sensing.registries import MODELS, DATASETS, LOSSES, METRICS, DISTILLERS, PREPROCESSORS
import kd_sensing.models
import kd_sensing.data
import kd_sensing.distillation
import kd_sensing.evaluation
import kd_sensing.preprocessing

print(MODELS.list())
print(DATASETS.list())
```

## Add a Model

```python
from kd_sensing.registries import MODELS

@MODELS.register("my_image_student")
class MyImageStudent:
    def __init__(self, feature_size, num_classes, gru_params):
        ...

    def forward(self, image_batch):
        return logits, input_features, output_features
```

Then reference it in YAML:

```yaml
model:
  student:
    type: my_image_student
    feature_size: 64
    num_classes: 64
    gru_params: [64, 64, 1]
```

Image models receive `[B, T, 1, 224, 224]`. Radar models receive `[B, T, 2, 128, 64]`. GPS models receive GPS-Rel-Polar tensors shaped `[B, T, 3]`. Fusion models receive only the tensors listed in `modalities`, using keyword inputs `image_batch`, `radar_batch`, and `gps_batch`. Models must return `(logits, input_features, output_features)`.

Built-in GPS model names follow the same teacher/student pattern as image and radar:

```yaml
model:
  student:
    type: gps_student
    gps_input_size: 3
    feature_size: 64
    num_classes: 64
    gru_params: [64, 64, 1]
```

Fusion modality selection is configured on both teacher and student:

```yaml
model:
  teacher:
    type: fusion_teacher
    modalities: [image, radar, gps]
    gps_input_size: 3
    feature_size: 64
    num_classes: 64
    gru_params: [64, 64, 2]
  student:
    type: fusion_student
    modalities: [image, radar, gps]
    gps_input_size: 3
    feature_size: 64
    num_classes: 64
    gru_params: [64, 64, 1]
```

## Add a Dataset

```python
from kd_sensing.registries import DATASETS
from torch.utils.data import Dataset

@DATASETS.register("my_dataset")
class MyDataset(Dataset):
    def __getitem__(self, idx):
        return {
            "image": image,
            "radar_ra": radar_ra,
            "radar_da": radar_da,
            "gps": gps,
            "input_beam": input_beam,
            "target_beam": target_beam,
        }
```

The engine expects the field names above for enabled modalities. GPS is optional for old image/radar configs, but GPS-only and GPS-enabled fusion configs require `gps`.

Scenario 9 GPS preprocessing supports one public `gps_feature_mode` value:

- `relative_polar`: `[dist, sin_theta, cos_theta]`

The dataset rejects `raw`, `utm`, `relative`, `motion`, and `motion_smooth` in this change. Those modes were used only for offline ablation selection and are not supported as maintained training entry points.

## Add a Loss, Metric, Distiller, or Preprocessor

Use the matching registry:

```python
from kd_sensing.registries import METRICS

@METRICS.register("my_metric")
class MyMetric:
    def __call__(self, outputs, labels):
        ...
```

Distillers receive student logits/features, teacher logits/features, labels, and current alpha. Preprocessors can expose a `run()` method and be invoked by `scripts/preprocess.py`.

## Error Handling

Unknown names, duplicate names, and missing constructor parameters raise `RegistryError` with the registry name and available component names.
