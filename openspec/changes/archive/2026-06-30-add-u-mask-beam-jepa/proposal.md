## Why

当前项目已有模块化多模态模型、difficulty/reliability metadata 和 JEPA 诊断基础，但还没有一个面向“任意模态子集缺失/损坏”的可训练 beam prediction 主模型。U-MaskBeamJEPA 将把用户给出的第一版研究设想收敛到现有 `src/kd_sensing`、registry、batch/runtime 和 OpenSpec 边界内，形成可审计、可 smoke 训练的最小实现方案。

## What Changes

- 新增 `u_mask_beam_jepa` whole-model exception，聚合模态 encoder latent、full-modal teacher、set context encoder、Gaussian JEPA predictor、reliability-gated fusion 和 beam head。
- 新增 U-MaskBeamJEPA 损失扩展：beam CE、teacher CE 和 Gaussian latent NLL，支持 ablation 关闭 teacher/JEPA/uncertainty 分支。
- 新增训练时 missing modality mask helper，并复用现有 canonical modality 与 difficulty metadata 语义；不新增伪模态名称。
- 新增 opt-in 配置与 smoke/ablation 配置，支持 eval 指定 missing pattern，并记录 reliability/global uncertainty 诊断指标。
- 不新增根目录 `train_u_mask_beam_jepa.py`、旧式 `models/`/`losses/` 顶层目录或兼容聚合层；实现落在 `src/kd_sensing` 的现有 owner 边界。

## Capabilities

### New Capabilities

- `u-mask-beam-jepa`: 缺失模态鲁棒 U-MaskBeamJEPA 模型、损失、mask 采样、配置接入、诊断指标和 ablation 行为。

### Modified Capabilities

- 无。现有 `model-architecture-extension-contract`、`modality-contracts`、`observability-aware-fusion`、`component-registry` 和 `first-class-prediction-tasks` 作为本 change 的约束，不改变其通用需求。

## Impact

- 代码：`src/kd_sensing/models/`、`src/kd_sensing/losses/`、`src/kd_sensing/data/`、`src/kd_sensing/engine/`、`src/kd_sensing/registries.py`、`configs/`、`tests/`。
- API：新增模型注册名 `u_mask_beam_jepa` 和 opt-in loss/objective 配置；普通 baseline 不需要传入 missing mask 或 reliability metadata。
- 数据与产物：不读取真实 `dataset/` 作为测试前提，不写入 tracked checkpoint/cache/log；训练输出继续走 ignored `outputs/`、checkpoint 和 metrics 现有路径。
- 依赖：不新增第三方依赖，只使用 PyTorch 和项目现有工具链。
