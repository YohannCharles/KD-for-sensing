# Extension Guide

The project uses lightweight registries. A config selects a component by `type`, and the remaining fields are passed to that component constructor.

## Inspect Available Components

```python
from kd_sensing.registries import MODELS, DATASETS, LOSSES, METRICS, PREPROCESSORS
from kd_sensing.registries import import_default_components

import_default_components()

print(MODELS.list())
print(DATASETS.list())
```

Importing registry objects is intentionally lightweight and does not import built-in datasets or models.
Call `import_default_components()` before inspecting built-in component lists or building a built-in
component. Custom components still need their defining module imported before the registry can see them.

## Add a Model

Most new supervised or adaptation baselines should use the modular model path. If the baseline only
changes a modality encoder, projector, representation/fusion core, or task head, express it with
`modular_sequence` config and the matching subcomponent registry instead of adding a new whole model.

Config-only baselines can be added by changing YAML:

```yaml
model:
  primary:
    type: modular_sequence
    modalities: [image, gps]
    d_model: 64
    num_classes: 64
    encoders:
      image:
        type: resnet18_imagenet_rgb
        pretrained: true
        freeze_backbone: true
      gps:
        type: gps_mlp
        gps_input_size: 3
        output_dim: 64
    representation_core:
      type: early_concat_gru
      d_model: 64
      hidden_size: 64
    heads:
      beam:
        type: beam_head
```

Component baselines register only the replaceable part:

```python
from kd_sensing.registries import ENCODERS

@ENCODERS.register("my_image_encoder")
class MyImageEncoder:
    def __init__(self, output_dim):
        ...

    def forward(self, image_batch, **metadata):
        ...

    def training_strategy_metadata(self):
        return {
            "architecture_category": "component_baseline",
            "component_role": "encoder",
            "uses_external_checkpoint": False,
            "freeze_policy": "none",
            "consumes_reliability_metadata": False,
        }
```

Then reference it in YAML:

```yaml
model:
  primary:
    type: modular_sequence
    modalities: [image]
    encoders:
      image:
        type: my_image_encoder
        output_dim: 64
```

Image models receive RGB/ImageNet tensors shaped `[B, T, 3, 224, 224]`. Radar models receive
`[B, T, 2, 128, 64]`. These image and radar sizes are structural constraints for the built-in
models; changing them requires updating fixed FC inputs or the radar branch. GPS models receive
GPS-Rel-Polar tensors shaped `[B, T, 3]`. LiDAR models receive BEV tensors
shaped `[B, T, 3, H, W]`. Fusion models receive only the tensors listed in `modalities`, using keyword
inputs `image_batch`, `radar_batch`, `gps_batch`, and `lidar_batch`. Models must return
`(logits, input_features, output_features)` or a dict with `logits`.

Image preprocessing is selected with `data.dataset.image_profile`. The default is `rgb_imagenet`,
paired with the `resnet18_imagenet_rgb` encoder for RGB street-view experiments such as Scenario 31-34.
The RGB profile produces `[B, T, 3, 224, 224]` ImageNet-normalized tensors and does not read or write an image cache.

The default image strong config uses `modular_sequence` with the ImageNet-pretrained
`resnet18_imagenet_rgb` encoder. Radar/GPS/LiDAR/mmWave single-modality configs also use
`modular_sequence`, with `radar_cnn`, `gps_mlp`, `lidar_cnn`, or `mmwave_mlp` selected under
`model.primary.encoders`. Image-containing canonical fusion strong configs also use the ResNet-18
image profile; other fusion configs may use their own encoder depth or lightweight branch. Canonical fusion virtual configs use
`strong` and `lightweight`; `experiment.name` and `output.run_name`
match the config stem. The trainable main model is always `model.primary`. Public radar config names are
`configs/radar/strong.yaml` and `configs/radar/lightweight.yaml`; the corresponding Python classes may keep older internal names,
but the registry names `radar_strong` and `radar_lightweight` are removed guards.
Checkpoint loading is strict by default; set `checkpoint.strict_load: false` only when intentionally
inspecting a partially compatible checkpoint, and check the reported missing/unexpected keys.

TinyViT is available only as an opt-in image encoder component; default image and image+GPS configs
continue to use `resnet18_imagenet_rgb`. Use the existing config and override the image encoder:

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/image/supervised.yaml \
  -o model.primary.encoders.image.type=tinyvit_5m_22k_rgb
```

Available TinyViT encoder names are:

- `tinyvit_5m_scratch_rgb`: TinyViT-5M, random initialization, no checkpoint load or download.
- `tinyvit_5m_22k_rgb`: TinyViT-5M with ImageNet-22k distill checkpoint loading.
- `tinyvit_11m_scratch_rgb`: TinyViT-11M, random initialization, no checkpoint load or download.
- `tinyvit_11m_22k_rgb`: TinyViT-11M with ImageNet-22k distill checkpoint loading.

TinyViT follows the same `rgb_imagenet` input profile as ResNet-18: `[B, T, 3, 224, 224]`,
ImageNet-normalized RGB frames, and no encoder-side resize. The 22k variants first use
`checkpoint_path` when provided; otherwise, with `allow_download=true`, they use the upstream
`wkcn/TinyViT-model-zoo` torch hub URL/cache path. The loader filters ImageNet classification
`head.*` weights and non-persistent `attention_bias_idxs` buffers, and records checkpoint provenance in
`training_strategy_metadata()`. TinyViT freezes the backbone by default, so only the projection and
downstream modular components train unless `freeze_backbone: false`, `unfreeze_stages`, or
`unfreeze_last_n_stages` is configured. TinyViT is not a KD or distillation training workflow in this
project; it is only a frame-level image encoder component, and no TinyViT weights, cache files,
checkpoints, logs, or training outputs should be committed.

Use the package architecture summary entry to inspect parameter counts and component roles without
training or reading a real dataset:

```bash
conda run -n kd_mm_beam python -m kd_sensing.cli.model_architecture_summary \
  --config configs/image/supervised.yaml \
  --format markdown
```

The maintained human-readable registry inventory is `docs/model_architecture_inventory.md`; update that
page whenever adding or retiring a model, encoder, projector, representation core, or head.

TinyViT override preflight is useful before launching a run, especially when reusing ResNet-specific
options:

```bash
conda run -n kd_mm_beam python -m kd_sensing.cli.model_architecture_summary \
  --config configs/image/supervised.yaml \
  -o model.primary.encoders.image.type=tinyvit_5m_scratch_rgb \
  -o model.primary.encoders.image.unfreeze_stages='[layer4]' \
  --no-build \
  --format json
```

JEPA visual sweep candidates can be rendered with the same fields used by startup summaries:

```bash
conda run -n kd_mm_beam python -m kd_sensing.cli.model_architecture_summary \
  --sweep-manifest configs/diagnostics/cnn_hybrid_jepa_visual_prior_sweep_manifest.yaml \
  --variant-id patch14_stage1_gps_query \
  --format csv
```

Built-in GPS configs follow the same modular strong/lightweight pattern as image and radar:

```yaml
model:
  primary:
    type: modular_sequence
    modalities: [gps]
    gps_input_size: 3
    feature_size: 64
    d_model: 64
    num_classes: 64
    encoders:
      gps:
        type: gps_mlp
    representation_core:
      type: single_gru
    heads:
      beam:
        type: beam_head
```

Fusion modality selection is configured on `model.primary.modalities` and may be mirrored by top-level
`model.modalities` for validation. The fixed modality contract lives in
`kd_sensing.modalities`; it defines order, dataset flags, sample keys, fusion input keys, default dataset
fields, default model fields, cache support, and normalization artifact names.

For image-containing canonical strong/lightweight configs, the image branch is represented through a
modular encoder entry:

```yaml
model:
  primary:
    type: modular_sequence
    modalities: [image, radar, gps]
    image_profile: rgb_imagenet
    encoders:
      image:
        type: resnet18_imagenet_rgb
        pretrained: true
        weights: DEFAULT
        freeze_backbone: true
        unfreeze_stages: [layer4]
```

Built-in fusion canonical configs cover the slug set generated in fixed order
`image -> radar -> gps -> lidar -> mmwave`:

```text
image_radar, image_gps, image_lidar, radar_gps, radar_lidar, gps_lidar
image_radar_gps, image_radar_lidar, image_gps_lidar, radar_gps_lidar
image_radar_gps_lidar

plus every two-, three-, four-, and five-modality combination that includes
mmWave, such as image_mmwave, radar_mmwave, gps_mmwave, lidar_mmwave,
image_radar_mmwave, and image_radar_gps_lidar_mmwave
```

Each slug has loadable canonical virtual paths `<slug>_strong.yaml` and
`<slug>_lightweight.yaml`.
These paths are generated by the config loader when no entity YAML exists; if an entity YAML exists, it
takes precedence. Top-level `model.modalities` and `model.primary.modalities` must stay identical when both
are present. Generated configs
derive `use_gps`, `use_lidar`, `use_mmwave`, GPS defaults, LiDAR defaults, mmWave defaults, and model
input fields from the modality contract. Generated fusion lightweight configs use
`cls_token_transformer_fusion` by default; strong configs use `modular_sequence` with the current
modality encoders and `early_concat_gru`.
Use explicit early-concat or token-transformer YAML/overlay paths when a baseline should
not follow the default. Retired research-line paths and fusion `logits_kd` / `rkd` virtual aliases are rejected instead of being generated as virtual configs.
New fusion extensions should default to the supervised/adaptation mainline and must not reintroduce KD runtime without a new OpenSpec change.
When adding a modality, update `kd_sensing.modalities` first, then add dataset
columns/readers, batch preparation, model registration, diagnostic rendering, and focused tests.

### Whole-model Exceptions

Direct `@MODELS.register(...)` should be used only for a whole-model exception, not as the default
way to add a baseline. A whole-model exception needs an OpenSpec design reason explaining why the
behavior cannot be represented as a config-only baseline or an encoder/projector/core/head component.
The change must document the registry name, config entry, enabled modalities, forward inputs,
output contract, training strategy metadata, and focused tests.

Whole-model exceptions must still reuse `engine.batch`, `engine.runtime.forward_task_model`, and
`adapt_model_output`. They must provide `training_strategy_metadata()` or an equivalent run metadata
helper covering the model registry name, architecture category, enabled modalities, checkpoint reuse,
freeze policy, and reliability metadata consumption. Focused tests should cover registry build,
synthetic forward, output adaptation, metadata, and config loading.

Default LiDAR configs use the modular sequence encoder path:

```yaml
model:
  primary:
    type: modular_sequence
    modalities: [lidar]
    lidar_channels: 3
    feature_size: 64
    d_model: 64
    num_classes: 64
    encoders:
      lidar:
        type: lidar_cnn
        output_dim: 64
        lidar_channels: 3
    representation_core:
      type: single_gru
      d_model: 64
      hidden_size: 64
      num_layers: 1
```

LiDAR BEV inputs are produced from `lidar1..lidarN` sequence CSV columns. The default BEV channels are height, intensity, and density. Default LiDAR baseline configs enable train-split streaming stats normalization, parameterized BEV cache directories, and runtime quality diagnostics. The built-in reader supports `.mat`, `.npy` point arrays, ASCII PCD, and numeric text/CSV point files. Binary PCD is intentionally not supported by default; convert it to ASCII PCD or `.npy` before training.

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
            "lidar": lidar,
            "input_beam": input_beam,
            "target_beam": target_beam,
        }
```

The engine expects the field names above for enabled modalities. GPS and LiDAR are optional for old image/radar configs, but GPS-only, LiDAR-only, and enabled fusion configs require the corresponding fields.
`target_beam` must contain only future labels `[t+1, ..., t+num_pred]`; the last historical beam stays in `input_beam` and is not part of the training label.

New dataset code should import transform helpers from `kd_sensing.data.transform_ops.image`,
`radar`, `gps`, `lidar`, `mmwave`, `io`, `cache`, or `normalization`. The old
top-level transform aggregation path has been removed; internal code should import the narrow modules directly.

Engine construction helpers are split by responsibility:

- `kd_sensing.engine.modality_resolution`: enabled modality inference and dataset flag conflict checks.
- `kd_sensing.engine.cache_policy`: high-level cache policy validation and dataset knob injection.
- `kd_sensing.engine.data_factory`: dataset and DataLoader construction.
- `kd_sensing.engine.normalization_artifacts`: GPS/LiDAR/mmWave scaler save and load.
- `kd_sensing.engine.run_metadata`: split, cache, and throughput metadata.
- `kd_sensing.engine.runtime`: shared batch normalization, label preparation, task input preparation,
  model forward, model-output adaptation, AMP helpers, and future-slot selection.
- `kd_sensing.engine.training_extensions`: lifecycle hooks for training methods that need extra losses,
  gradient post-processing, or epoch diagnostics.
- `kd_sensing.engine.optim`: model/loss/metric, optimizer, scheduler, and device construction.

The old builder aggregation module has been removed. Import the responsibility-specific module instead.

## Add a Training Method

Training methods that need extra losses or diagnostics should be implemented as engine training
extensions instead of adding method-specific branches to `engine.trainer`. The public `train(cfg)` entry
point remains the lifecycle owner: it builds dataloaders, models, loss objects, optimizer,
scheduler, checkpoint payloads, TensorBoard scalars, history, and final config snapshots. Method modules
plug into that lifecycle through `TrainingExtension` hooks:

- `setup(context)`: construct method runtime objects and return load summaries.
- `before_epoch(context, state, epoch=...)`: reset method epoch state or diagnostics accumulators.
- `before_forward(context, state, batch, labels, epoch=...)`: provide force masks or other method controls
  for the shared forward runtime.
- `compute_base_loss(context, state, batch_state)`: optionally replace the standard base loss.
- `after_forward(context, state, batch_state)`: add extra losses and scalar diagnostics.
- `after_backward(context, state, batch_state)`: apply gradient post-processing before clipping/optimizer step.
- `after_epoch(context, state, epoch=...)`: emit epoch diagnostics.

Use `BatchState`, `ForwardControls`, `BaseLossResult`, `LossBundle`, and `EpochDiagnosticsAccumulator`
from `kd_sensing.engine.training_extensions` rather than inventing per-method logging structures.

Do not put method-specific auxiliary forwards or method-specific scalar aggregation back into
`engine.trainer`. Add a method extension module and a
focused architecture-boundary test instead.

## Shared Forward Runtime

Training, validation, viewer predictions, subset diagnostics, and supported method runtimes should call
`engine.runtime.prepare_task_inputs`, `forward_task_model`, or `run_model_step`.
Those helpers centralize:

- legacy tuple/dict batch normalization
- future-label preparation
- task-specific modality input tensors
- `force_modality_mask` forwarding
- `ModelOutput` adaptation
- future-slot selection

When adding or changing a modality input contract, update `engine.batch`/`engine.runtime` and the runtime
tests first. Avoid copying task branches into validator, viewer, or method extension code.

## Prediction Objectives and Evaluation

Prediction task metadata lives in `kd_sensing.engine.prediction_objectives`. Add or modify an objective
there first: target requirements, output requirements, primary loss, default metric, metric direction,
early-stopping aliases, available validation metrics, history fields, TensorBoard scalar mappings, and
runtime metadata are all part of the objective spec. Training, validation, evaluation reports, checkpoints,
and final configs consume that metadata instead of keeping local metric tables.

Validation-like flows should call `kd_sensing.engine.evaluation_pass.run_evaluation_pass`. The shared pass
owns batch preparation, model forward, objective loss calculation, auxiliary metric collection, Top-K/DBA
aggregation, degradation diagnostics, available metrics, objective metadata, and enabled modality metadata.
`engine.validator.validate`, force-mask subset validation, and standalone checkpoint evaluation are wrappers
around that pass. Do not add a second validation loop to `validator.py`.

DeepSense6G target construction is split from the dataset coordinator. `deepsense6g_targets.py` provides
beam-compatible auxiliary targets for occlusion, position, and multitask objectives while preserving sample
field names and tensor shapes. `deepsense6g_loaders.py` is the modality loader boundary for image, radar,
GPS, LiDAR, and mmWave inputs. Disabled targets and modalities should not initialize or read their resources.

Canonical virtual config generation is recipe driven under `kd_sensing.config.canonical_recipes`. Base
fusion mode defaults, objective overlays, and supported advanced overlays are table entries; keep
`config/canonical.py` as path parsing and recipe application glue.

`kd_sensing.models` is a lazy export package. Keep public names in its export mapping, but import concrete
implementation modules directly when editing model internals. A plain `import kd_sensing.models` should not
pull in fusion, GPS, LiDAR, mmWave, image encoder, or radar implementation modules.

## Retired Viewer Manifest

The repository-level Gradio viewer, viewer manifest export CLI, `kd-sensing-visualize-modalities` alias, and
`viewer_manifest_*` helper modules are retired. Do not add compatibility stubs or new helper modules under
the old viewer names. Current diagnostic work belongs in JEPA visual analysis, GPS shortcut benchmark, or an
explicitly specified non-viewer diagnostic owner.

## Advanced Fusion Overlays

Advanced fusion recipes can be loaded through virtual overlay paths under `configs/fusion/overlay_*.yaml`.
The loader composes a shared five-modality fusion base with a method overlay and an optional ablation
overlay. Existing physical YAML files still take precedence over generated overlays.

Current overlay recipes include:

```text
overlay_multitask_occlusion_position
```

Use command-line overrides for scene selection, for example `data.dataset.scene=9`, rather than copying
method configs per scene. Training still writes the fully resolved `final_config.yaml`, so generated
overlay experiments remain reproducible.

Scenario 9 GPS preprocessing supports one public `gps_feature_mode` value:

- `relative_polar`: `[dist, sin_theta, cos_theta]`

The dataset rejects `raw`, `utm`, `relative`, `motion`, and `motion_smooth` in this change. Those modes were used only for offline ablation selection and are not supported as maintained training entry points.

## Add a Loss, Metric, or Preprocessor

Use the matching registry:

```python
from kd_sensing.registries import METRICS

@METRICS.register("my_metric")
class MyMetric:
    def __call__(self, outputs, labels):
        ...
```

Preprocessors can expose a `run()` method and be invoked by `scripts/preprocess.py`.

## Error Handling

Unknown names, duplicate names, and missing constructor parameters raise `RegistryError` with the registry name and available component names.

## Local Artifacts

`dataset/` is a local data input. Tracked `All_models/*.pth` files are historical reproduction artifacts
and are not consulted by default checkpoint resolution. New checkpoints from training, evaluation,
diagnostics, or cache generation should stay under ignored paths such as `outputs/`, `logs/`, cache folders,
or files matched by `*.pth` / `*.pt` / `*.ckpt`. Do not include generated artifacts in source changes unless
a task explicitly asks for fixture data or documentation updates.
