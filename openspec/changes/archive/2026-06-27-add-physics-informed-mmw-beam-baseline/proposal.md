## Why

当前 MMW/CSI/fusion 工作流已经能做配置驱动 beam prediction，但物理监督主要停留在 beamspace label、CSI hardening 和诊断层，缺少一个可训练、可审计的“感知特征 -> 主导路径参数 -> 可微信道重构 -> beam 选择”闭环 baseline。这个 change 用最小新增 surface 把 PINN/physics-informed 思路落到现有 `src/kd_sensing` 架构里，避免另起一套 `data/ models/ train.py` 入口。

## What Changes

- 新增一个 MMW physics-informed beam prediction baseline，面向 Multimodal-Wireless/MMW Town10 数据与可派生 CSI/path/beam-power 监督。
- 新增可微物理模块：ULA array response、path-parameter channel synthesizer、complex tensor helper、beam codebook scoring 和 physics loss bundle。
- 新增 MMW physics supervision adapter，把现有 `csi`、`beamspace_power_label`、`path_descriptor`/path payload 或等价字段统一成模型/loss 可消费的标准 batch auxiliary 字段；字段缺失时用 mask 跳过对应 loss。
- 新增 `PINNMultimodalBeamModel` whole-model exception，复用现有 batch/runtime、registry、配置和训练入口，输出继续兼容 `adapt_model_output`，额外 diagnostics 承载 `direct_logits`、`physics_logits`、`h_hat` 和 `path_hat`。
- 新增 physics-aware loss/metric 扩展，支持 beam CE、beam-power KL/MSE、CSI reconstruction NMSE、path parameter smooth-L1/complex loss、array/beam consistency 和可选轻量 alignment loss。
- 新增 canonical/debug/ablation 配置，覆盖 CSI-only、RGB/image-only、普通 fusion no-physics、hybrid physics、no-CSI-reconstruction、no-path-loss、no-array-consistency 和 no-physics-head。
- 新增包内 dataset inspection/shape smoke 能力，首次运行记录 RGB/CSI/beam/path/subcarrier/antenna/beam 数等关键 shape，但不新增根目录 Python 脚本。
- 更新文档和模型架构 inventory，明确该 baseline 是 physics-aware/diagnostic 或 source-supervised baseline；使用 CSI/path/beam-power 作为模型输入或 target-side 训练监督的 run 不进入现有 MMW sensor-assisted 主结论集合。
- 不新增旧式根目录 `train.py`、`evaluate.py`、`scripts/inspect_dataset.py` thin alias，不复制通用训练循环，不提交本地数据、cache、checkpoint 或真实数据 inspection 输出。

## Capabilities

### New Capabilities

- `physics-informed-mmw-beam-baseline`: 定义 MMW physics-informed baseline 的数据监督、模型闭环、物理 loss、指标、配置、metadata、inspection 和验证契约。

### Modified Capabilities

- `model-architecture-extension-contract`: 增加该 PINN baseline 作为 whole-model exception 的审计要求和模型架构摘要覆盖。
- `mmw-sensor-assisted-beam-prediction`: 明确 physics-informed run 与 sensor-assisted 主结论 eligibility 的边界，防止 CSI/path/beam-power oracle 泄漏。
- `csi-channel-data`: 扩展 CSI 监督字段在 physics reconstruction 中的消费规则、shape 适配和 mask 行为。
- `beamspace-physical-labels`: 扩展 beamspace power/path physical label 可作为 physics loss/diagnostic 的监督来源，并保持 target-domain leakage boundary。
- `experiment-workflow`: 增加 physics-informed MMW 配置、包内 inspection smoke 和 ablation workflow 的入口契约。

## Impact

- 代码：`src/kd_sensing/models/`、`src/kd_sensing/models/physics/`、`src/kd_sensing/losses/`、`src/kd_sensing/evaluation/`、`src/kd_sensing/data/datasets/mmw*.py` 或窄 adapter、`src/kd_sensing/cli/`、`src/kd_sensing/registries.py`、`src/kd_sensing/engine/model_output.py` 相关适配路径。
- 配置：新增 `configs/fusion/physics_informed_mmw_*.yaml` 和最小 ablation overlay；不新增仓库根 `configs/default.yaml`。
- 文档：README 只给 quickstart/入口；`docs/mainline_model_catalog.md`、`docs/experiment_protocols.md`、`docs/result_claims_registry.md`、`docs/experiment_matrix.md` 和 `docs/model_architecture_inventory.md` 记录实验口径与 claim 状态。
- 测试：registry build、synthetic forward/loss/backward、complex physics autograd、MMW adapter missing-field mask、config load、CLI help/inspection smoke、architecture boundary 和 model architecture summary focused tests。
- 依赖：不新增第三方依赖；如实现 path matching，第一版只按 gain magnitude 排序，Hungarian matching 仅保留后续扩展点。
