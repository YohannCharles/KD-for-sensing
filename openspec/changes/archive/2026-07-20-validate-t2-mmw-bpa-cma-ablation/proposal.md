## Why

T2 在 MMW 主实验上优于 AMBER-Full，但现有 PCA/t-SNE 不能给出可归因的机制证据。需要在完全相同的数据、训练预算和评估掩码上，将 Beam Prototype Alignment（BPA）的辅助目标、beam 拓扑假设和 prototype head 分别做受控消融，回答性能来自哪里，而不是仅展示最终指标更高。

## What Changes

- 新增 MMW T2 配对消融矩阵：完整 T2、仅关闭 BPA、将 BPA 替换为 AMBER 风格跨 batch CMA、线性 beam 距离、classifier head，以及 classifier head + CMA。
- 新增与 AMBER 论文公式语义一致的“池化特征 CMA 类比目标”：可用模态为 anchor，同样本融合特征为正样本，batch 内其他样本融合特征为负样本；对带放回采样造成的重复 `sample_id` 使用多正样本处理。
- 固定 15 个“场景×天气”域、40 epoch、相同 seed、相同缺失 curriculum 和相同评估掩码；复用现有完整 T2 checkpoint，不改变数据集或主模型结构。
- 分离三类因果问题：`T2-NoBPA` 只测 BPA 辅助目标，`T2-BPA2CMA` 只测 BPA 与 CMA 的目标替换，`T2-CLS`/`T2-CLS-CMA` 测完整去 prototype package；不把这些结果混写成同一种消融。
- 对 circular 与 linear BPA target 结果增加精确端点 beam `{0,63}`、近端点 `{62,63,0,1}` 和内部 beam 分层统计，以检验循环先验的收益是否确实来自码本索引边界。
- 输出三 seed 均值/标准差、paired delta、缺失率曲线和可直接用于论文的消融表与图；所有 claim 必须标注该 CMA 是 T2 现有池化特征上的 objective analogue，不宣称复现完整 AMBER Class-Former。

## Capabilities

### New Capabilities

- `mmw-t2-bpa-cma-ablation`: 规定 T2 在 MMW 上的 BPA、CMA、beam 拓扑和 prototype head 配对消融协议、训练契约、评估切片与 claim 边界。

### Modified Capabilities

无。

## Impact

- 影响 `src/kd_sensing/losses/` 与 U-Mask Beam JEPA 配置解析：新增可选、默认关闭的 AMBER 风格 batchwise CMA 辅助目标。
- 影响 MMW launcher、evaluator 和汇总脚本：新增消融方法标签、配置覆盖、端点/内部切片及多 seed 汇总。
- 新增相关单元测试、OpenSpec artifacts 和本地训练/评估产物；不新增第三方依赖，不改变默认训练行为，不提交 checkpoint、日志或数据。
