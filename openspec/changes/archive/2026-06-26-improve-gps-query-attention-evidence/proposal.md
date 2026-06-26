## Why

当前 GPS-query attention overlay 视觉上有热点，但语义不够明确：它展示的是 GPS 条件 query 对视觉 patch token 的读取权重，而不是 beam 预测的因果归因。现有报告已经提醒 attention 只能作为解释性诊断，但还缺少能验证其可信度的遮挡/反事实检查、稳定归一化口径和模型侧 attention 聚合 provenance。

本 change 的目标是把 attention 图从“单张好看的热力图”升级为可审计的 evidence package：主证据仍是 paired ablation 和扰动指标，attention 只在通过 faithfulness 检查时作为解释性支持；若检查不通过，报告必须明确降级。

## What Changes

- 将 GPS-query attention 图的默认解释口径改为 `token_read_map`：明确表示 query-to-patch token readout，不单独称为 causal explanation 或 attribution。
- 增加 attention faithfulness 诊断：对 top-attention patch、low-attention patch 和随机 patch 做 deterministic 遮挡/替换，比较 logits、DBA、Top-k 或目标 beam margin 的变化。
- 扩展 evidence package 的 claim gate：将 faithfulness 结果、attention 可用性、query diversity、effective patch count 和 paired delta 一起纳入 `supported`、`exploratory`、`insufficient` 或 `blocked` 判定。
- 改进 overlay 产物 metadata：记录归一化方式、底图来源、query/time/head 聚合方式、token grid、Top-p patch 覆盖率和是否使用模型输入图兜底。
- 扩展 JEPA visual analysis 输出：除现有 per-sample minmax overlay 外，支持全局/数据集级共享尺度或至少在 manifest 中记录不可比原因，并输出遮挡诊断表和简洁图表。
- 扩展 GPS-query 类 pooler 诊断 metadata：记录 attention 是否跨 head 平均，并允许 opt-in 导出 per-head attention 或分支 attention summary；训练主 forward 语义不改变。
- 不新增默认训练入口，不恢复旧 viewer/Gradio/viewer manifest，不把 attention 图单独包装成有效性证明。
- 无 **BREAKING** 变更；所有新增诊断均为 opt-in 或向后兼容的附加输出。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `gps-query-effectiveness-visualization`: 增加 attention faithfulness、token-read 解释边界、claim gate 判定和 evidence 输出契约。
- `jepa-visual-analysis-suite`: 增加 JEPA visual analysis 中 attention 遮挡诊断、归一化/聚合 provenance 和报告降级要求。
- `jepa-downstream-extensibility`: 增加 GPS-query 类 pooler 的 attention aggregation metadata 和可选 per-head/分支 attention 诊断契约。

## Impact

- 主要影响 `src/kd_sensing/diagnostics/jepa_visual_analysis.py` 与 `src/kd_sensing/diagnostics/gps_query_evidence.py` 的离线诊断、manifest/report/table 写出。
- 小范围影响 `src/kd_sensing/models/jepa_downstream.py` 的诊断 metadata；不改变默认模型输出、不改变训练损失、不要求 checkpoint schema 迁移。
- 新增或更新 synthetic/mock tests，重点覆盖 attention shape 解析、遮挡 ranking、claim gate 降级、manifest provenance 和没有 attention 时的安全降级。
- 所有真实图表、CSV、cache 和 case payload 仍写入 ignored 的 `outputs/` 或用户显式指定的本地产物目录，不提交真实运行产物。
