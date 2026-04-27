# KD for Sensing

This repository is now organized as an installable `src/kd_sensing` package with config-driven training, evaluation, and preprocessing entry points.

## Install

```bash
conda activate kd_mm_beam
pip install -e .
```

The package import is side-effect free:

```bash
python -c "import kd_sensing"
```

## Structure

```text
configs/
  image/          # image-only no-KD, logits KD, RKD configs
  fusion/         # image+radar no-KD, logits KD, RKD configs
  preprocess/     # CSV/radar/sequence preprocessing configs
scripts/
  train.py
  evaluate.py
  preprocess.py
src/kd_sensing/
  cli/
  config/
  data/
  distillation/
  engine/
  evaluation/
  models/
  preprocessing/
  utils/
```

Large data and pretrained weights stay in their existing locations:

- `dataset/`
- `All_models/`

Relative paths in configs are resolved from the project root, so commands can be launched from subdirectories.

## Train

```bash
python scripts/train.py --config configs/image/no_kd.yaml
python scripts/train.py --config configs/image/logits_kd.yaml
python scripts/train.py --config configs/image/rkd.yaml

python scripts/train.py --config configs/fusion/no_kd.yaml
python scripts/train.py --config configs/fusion/logits_kd.yaml
python scripts/train.py --config configs/fusion/rkd.yaml
```

Override config values with dotted keys:

```bash
python scripts/train.py --config configs/image/rkd.yaml training.epochs=1 data.dataset.portion=0.05
```

Outputs are written under `outputs/<run_name>/` and include:

- `final_config.yaml`
- `checkpoints/last.pth`
- `checkpoints/best.pth`
- `metrics.json`
- `train_log.json`
- `training_outputs.npz`
- training curves

## Evaluate

```bash
python scripts/evaluate.py --config configs/image/no_kd.yaml --weights All_models/ImageTeacher_noKD.pth
python scripts/evaluate.py --config configs/fusion/rkd.yaml --weights All_models/BothStd_RKD.pth
```

Evaluation writes metrics and `test_report.json` to the configured output directory.

## Preprocess

```bash
python scripts/preprocess.py --config configs/preprocess/radar_ra.yaml
python scripts/preprocess.py --config configs/preprocess/radar_da.yaml
python scripts/preprocess.py --config configs/preprocess/sequences_ra.yaml
```

## Breaking Change

The old top-level entry scripts were removed. Use the new commands instead:

| Old command | New command |
| --- | --- |
| `python train_image.py ...` | `python scripts/train.py --config configs/image/<mode>.yaml ...` |
| `python train_both.py ...` | `python scripts/train.py --config configs/fusion/<mode>.yaml ...` |
| `python test_model_image.py ...` | `python scripts/evaluate.py --config configs/image/<mode>.yaml --weights <path>` |
| `python test_model_both.py ...` | `python scripts/evaluate.py --config configs/fusion/<mode>.yaml --weights <path>` |
| `python CSV_process.py ...` | `python scripts/preprocess.py --config configs/preprocess/radar_ra.yaml` |
| `python gen_data_seq.py ...` | `python scripts/preprocess.py --config configs/preprocess/sequences_ra.yaml` |

## Components

Built-in registries live in `kd_sensing.registries`:

- `MODELS`
- `DATASETS`
- `LOSSES`
- `METRICS`
- `DISTILLERS`
- `PREPROCESSORS`

See [docs/extension_guide.md](docs/extension_guide.md) for adding new components.

