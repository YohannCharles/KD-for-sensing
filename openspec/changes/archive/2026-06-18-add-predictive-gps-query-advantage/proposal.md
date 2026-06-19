## Why

当前 `JEPA GPS-query` 与 `Image ResNet+GPS` 在真实 P0-P5 evaluation 中只相差约千分位，说明单纯增加 GPS-query 的 `k_queries` 没有稳定拉开差距。问题不应通过替换主指标来掩盖，而应让 GPS-query 架构真正具备 JEPA 的预测式 latent 优势，并补充能检验 GPS-query 归纳偏置的严格 advantage 场景。

## What Changes

- 新增 opt-in 的 Predictive GPS-query++ downstream 架构方案：在现有 GPS-query pooler 基础上引入 content-query anchor、GPS-query residual、causal temporal latent predictor、reliability-aware gate 和 latent consistency diagnostics。
- 扩展 predictive robustness 评测：保留 P0-P5 作为主 claim 口径，同时增加 GPS-query advantage slice，用于评估视觉歧义、beam-offset-constrained wrong GPS、GPS async/low-rate 与 image degradation 的组合条件。
- 扩展 difficulty pipeline：新增或标准化 hard negative 构造，包括 visual-ambiguous peer、beam-offset-constrained GPS counterfactual、当前/历史图像 burst missing 与 GPS wrong/async 的组合扰动。
- 扩展 observability/reliability gate 契约：opt-in JEPA gate 可以消费 `image_valid_mask`、`image_observability_score`、`gps_valid_mask`、`gps_counterfactual_mask`、`gps_delay_steps` 等连续可靠性信号，但不得直接读取 condition id。
- 增加同协议实验配置与诊断输出：`Image ResNet+GPS`、当前 `JEPA GPS-query k=4`、Predictive GPS-query++ 必须在相同 split、history window、prediction horizon、seed、difficulty digest 和 metric profile 下比较。
- 不删除 P0-P5，不把旧 C/D 矩阵替代为主指标；C/D 或 CxD 仅作为机制诊断和 robustness sanity。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `jepa-downstream-extensibility`: 增加 Predictive GPS-query++ pooler/gate/temporal latent predictor 的配置、forward、diagnostics 和默认兼容要求。
- `predictive-jepa-robustness`: 增加 GPS-query advantage slice、P-hard 条件和严格可比较的 GPS-query++ vs ResNet/GPS-query baseline claim 口径。
- `modality-difficulty-pipeline`: 增加 visual-ambiguous hard negative、beam-offset-constrained wrong GPS 和组合扰动 metadata 契约。
- `observability-aware-fusion`: 增加 reliability-aware gate 对连续可靠性信号的消费要求，并明确 condition id 不能作为 gate 输入。

## Impact

- 受影响模型模块：`src/kd_sensing/models/jepa_downstream.py`、`src/kd_sensing/models/jepa.py`、`src/kd_sensing/models/modular.py`、相关 registry/default import 与 runtime metadata。
- 受影响数据与 difficulty 模块：`src/kd_sensing/data/difficulty/presets.py`、`src/kd_sensing/data/difficulty/operators/`、batch metadata 传递路径。
- 受影响配置与诊断：`configs/fusion/experiments/jepa_image_gps/` 下新增派生配置，`configs/diagnostics/` 下新增真实/烟测 manifest 或本地产物模板，诊断输出写入 ignored `outputs/analysis/...`。
- 受影响测试：JEPA pooler/gate synthetic forward、difficulty determinism/no-future-leak、config load characterization、benchmark manifest/metric comparability、architecture boundary tests。
- 产物边界：真实训练、checkpoint、P0-P5/CxD metrics 和 PNG/CSV 图表仍只写入 ignored `outputs/`、`logs/` 或 manifest 指定目录，不进入源码变更。
