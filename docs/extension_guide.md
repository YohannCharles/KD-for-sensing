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

Image models receive `[B, T, 1, 224, 224]`. Fusion models receive image plus radar `[B, T, 2, 128, 64]`. Models must return `(logits, input_features, output_features)`.

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
            "input_beam": input_beam,
            "target_beam": target_beam,
        }
```

The engine expects the field names above.

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

