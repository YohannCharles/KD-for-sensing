## Why

`add-predictive-gps-query-advantage` 的新 strict run 显示，Predictive GPS-query++ 在 clean/P0-P5 与 advantage slice 上均显著低于 `Image ResNet+GPS` 和 `JEPA GPS-query k=4`，说明继续在 latent query/predictor 上叠复杂度不能形成可靠收益。

下一步应转向更符合 beam prediction 文献和失败诊断的方案：把 GPS 用作几何 beam prior 与可校准可靠性证据，把 image/temporal branch 作为主感知信号，并用 DBA-aware supervised loss、teacher logit distillation 和 clean-first curriculum 保护 clean 性能。

## What Changes

- 新增 opt-in `Geometry-Prior Beam Fusion` 路线：从 GPS relative polar / relative Cartesian / angle calibration 生成 beam prior distribution 或 prior logits，并与 image/fusion logits 在 logit 层融合。
- 新增 geometry-aware GPS encoder/head 组件，优先作为 `modular_sequence` component baseline 实现，不新增旧入口、不复制训练循环、不恢复退役 GPS residual 路线。
- 增加 DBA-aware supervised objective：支持 beam topology/circular Gaussian/beam power soft label、distance-aware CE 或 ordinal/EMD-style loss，并保持 validation/evaluation 仍使用 hard `target_beam` 的 Top-K/DBA。
- 增加 teacher-guided stabilization：允许使用当前强 `Image ResNet+GPS` strict checkpoint 作为 teacher，对 candidate logits 做温度蒸馏或 rank/DBA-aware distillation；该项必须标记为 training stabilization，不得宣称为旧 KD research line 回归。
- 增加 clean-first robustness curriculum：训练必须先保证 clean/P0 性能，再逐步混入 P1-P5 和 GPS advantage perturbation，禁止只在单一 hard condition 上训练后直接升级 claim。
- 增加 uncertainty/reliability logit fusion：branch 输出 beam distribution 与 uncertainty/evidence，融合权重基于 GPS reliability、image observability、teacher agreement 或 entropy，而不是 condition id。
- 增加诊断与 claim gate：报告 GPS prior accuracy/entropy、prior-vs-image agreement、branch logit weights、teacher agreement、clean/P-suite/advantage margins，并在 clean regression 超阈值时阻止 claim upgrade。
- 所有真实 checkpoint、CSV、PNG、TensorBoard 和 runtime runner 产物仍写入 ignored `outputs/` 或 `logs/`，源码只提交配置、代码、测试和 OpenSpec/文档摘要。

## Capabilities

### New Capabilities

- `geometry-prior-beam-fusion`: 定义 GPS/几何 beam prior、logit-level fusion、teacher-guided stabilization、clean-first curriculum、strict claim gate 和 diagnostics 的行为契约。

### Modified Capabilities

- `configurable-multimodal-fusion`: 增加 geometry-prior fusion 作为 opt-in component baseline，并声明其配置、模态输入和 output contract。
- `model-architecture-extension-contract`: 明确该路线默认属于 component baseline；只有复现 BEV-Fusion 论文协议时才可走 workflow/paper reproduction。
- `soft-beam-label-training`: 扩展 beam topology / DBA-aware loss 的配置与日志命名，要求它保持 supervised beam smoothing 语义而不是旧 KD。
- `observability-aware-fusion`: 增加 logit-level uncertainty/evidence fusion，要求普通 baseline 可忽略 reliability metadata，且 condition id 只用于诊断分组。

## Impact

- 受影响模型模块：`src/kd_sensing/models/` 中的 GPS/geometry encoder、representation core 或 head 组件，`ModelOutput` adaptation 和 run metadata。
- 受影响 loss/objective：`src/kd_sensing/engine/prediction_objectives.py` 或相关 beam loss helper，新增 DBA-aware/beam topology supervised loss 与可选 teacher-guided stabilization。
- 受影响 batch/runtime：`src/kd_sensing/engine/batch.py`、`src/kd_sensing/engine/runtime.py` 只做 opt-in metadata 和 teacher logits 传递，不要求普通 baseline 接收新增字段。
- 受影响配置：`configs/fusion/experiments/jepa_image_gps/` 或新的 geometry-prior 配置，必须包含 strict H5/G2/F1、scene32-34、future=1、seed=17 的 smoke 与 real-run 配置。
- 受影响诊断：`src/kd_sensing/diagnostics/` 下增加 geometry-prior comparison/claim gate，输出 prior quality、branch weights、teacher agreement 和 strict comparison tables。
- 受影响测试：配置加载、component registry、synthetic forward、loss、metadata isolation、diagnostics aggregation、architecture boundary 和 focused evaluation tests。
