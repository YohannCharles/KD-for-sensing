## Why

当前 HiST-Beam / MMW 几何版已经具备可运行的 shared/private、geometry-aware、adapter、coarse prototype、LOSO 和 claim guard 闭环，但 prototype 仍以 coarse sector 或 private adapter 聚类为主，无法表达 beam-power profile 中的窄峰、宽峰、多径和能量分散等更稳定的无线传播形态。

`跨场景自适应方案_融合推理修改版.md` 指向的下一步应是增量升级：在不重写现有 MMW 数据准备、LOSO 和 HiST-Beam 主干的前提下，引入 radio-semantic label、shared radio prototype、target-private prototype bank 和可选 radio assignment 融合推理，使 V5 coarse prototype 与新的 radio-semantic prototype 形成可验证、可诊断的关键消融。

## What Changes

- 新增 Radio-Semantic Prototype HiST-Beam 能力：从 source beam-power profile 派生 `radio_semantic_label`，让 shared branch 预测无线传播语义，而不是只预测 `beam // group_size`。
- 新增 `RadioSemanticLabelBuilder` 契约，支持 `coarse` fallback、快速主方法 `peak_spread`，以及后续可选的 `kmeans_power`；无 beam power 时必须降级并记录 unavailable reason。
- 扩展 HiST-Beam 模型：增加 `radio_head(c)`、可选 `radio_embedding` 和 `beam_head([c, s*])` / `beam_head([c, s*, e_alpha])` 两种融合推理路径；radio prototype 不得直接输出 beam。
- 扩展 source prototype artifact：在 shared representation 空间保存 `mu_radio_c`、`count_radio`，保留 coarse prototype 仅作 V5 消融；source private prototype 默认不再作为 target private 强对齐目标。
- 扩展 target adaptation：新增 radio prototype assignment、target-private prototype bank `nu_radio_s`、EMA 更新、warmup 后的 target 内部 private clustering loss，以及 label-budget=0 的 leakage guard。
- 扩展 loss 与 metrics：新增 radio semantic CE、radio semantic accuracy、normalized received power、beam power loss dB、radio prototype coverage/confidence、target-private prototype diagnostics 和训练期 leakage 记录。
- 扩展 LOSO / few-shot sampling：当 target_adapt 中可合法使用 radio semantic label 时，few-shot 采样优先 radio-semantic 分层，其次 coarse sector / relative azimuth，最后 deterministic random fallback。
- 保持当前 MMW geometry-aware 和 DeepSense6G HiST-Beam 入口兼容；现有 `v5_adapter_proto` 继续表示 coarse/private prototype baseline，新 full method 使用显式 `v6_radio_proto` 或等价配置名，full fine-tuning baseline 在报告中作为 V7 对照。

## Capabilities

### New Capabilities

- `radio-semantic-hist-beam-adaptation`: 定义 radio-semantic label 构造、shared radio prototype、target-private prototype bank、radio-conditioned beam inference、leakage guard 和 V5/V6/V7 消融契约。

### Modified Capabilities

- `hist-beam-cross-scene-adaptation`: 将现有 adapter+prototype 变体扩展为 coarse prototype baseline 与 radio-semantic prototype full method 的可配置对比，并要求模型/loss/prototype/adaptation 输出 radio 相关 diagnostics。
- `cross-scene-loso-workflow`: 扩展 few-shot sampling、target adaptation 防泄漏和 summary 汇总字段，使 radio-semantic 分层、0-label 禁用 target radio label 训练、V5/V6/V7 对比可审计。
- `mmw-town10-dataset-preparation`: 扩展 MMW beam power / channel-derived metadata 的派生标签契约，支持在 prepared manifest 或 dataset runtime 中生成可追踪的 radio-semantic label 与 unavailable reason。

## Impact

- 影响模型：`src/kd_sensing/models/fusion/hist_beam.py` 的配置解析、forward 输出、beam head 输入维度、radio head / embedding 和 variant metadata。
- 影响数据与标签：`src/kd_sensing/data/datasets/mmw.py`、`src/kd_sensing/data/mmw/`、新增或扩展 radio semantic label builder，读取 beam power 但不把 CSI/channel 作为 sensing 输入模态。
- 影响训练与 loss：`src/kd_sensing/engine/hist_beam_losses.py`、`src/kd_sensing/engine/hist_beam_training.py`、batch label 搬运和 diagnostics。
- 影响 prototype / adaptation：`src/kd_sensing/engine/hist_beam_prototypes.py`、`src/kd_sensing/engine/hist_beam_adaptation.py`、LOSO execution 对 prototype artifact、trainable 参数和 leakage metadata 的记录。
- 影响评估与汇总：`src/kd_sensing/evaluation/hist_beam_outputs.py`、`src/kd_sensing/engine/hist_beam_loso_execution.py`、`src/kd_sensing/cli/hist_beam_loso.py` 和 quick validation conclusion。
- 影响配置与测试：新增 MMW radio semantic smoke / scenario LOSO 配置，扩展 `tests/test_hist_beam*.py`、`tests/test_mmw_town10_preparation.py`、LOSO 和 leakage guard 相关单测。
