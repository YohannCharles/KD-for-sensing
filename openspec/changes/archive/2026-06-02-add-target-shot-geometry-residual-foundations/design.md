## Context

当前仓库已经有 MMW Town10 manifest、group-safe sequence split、history-anchored residual、image-only legal probe 和 quick validation eligibility audit 等基础，但这些能力分散在特定实验路径中。附件中的研究路线需要先建立两个更通用的基础层：一是可复现的 5% target-shot source/target split，二是基于 BS/RSU-centric geometry 的 coarse beam 与 residual label。该基础层必须服务 MMW 多场景/多天气，也应尽量复用现有 DeepSense6G/Raymobtime 的 dataset runtime 和 split metadata 契约。

本变更只做数据、标签和诊断基础，不改变现有训练主线默认行为。所有项目相关 Python 命令、测试和诊断必须通过 `conda run -n kd_mm_beam <command>` 执行。

## Goals / Non-Goals

**Goals:**

- 提供配置驱动的 domain 定义与 5% target-shot split builder，支持 scenario、weather/condition、scenario+weather、town+scenario+weather。
- 将 target domain 拆分为 `target_labeled`、可选 `target_unlabeled` 和 `target_test`，并持久化 sample ids、indices、domain metadata、label histogram 和 leakage diagnostics。
- 提供 geometry-residual beam label 工具，覆盖 `beam_geo`、`geo_angle`、`geo_sector`、`beam_residual`、`residual_class`、circular distance 和 residual-to-absolute 还原。
- 让 dataset/target provider 在 `label_space.type: geometry_residual` 时按需暴露 geometry/residual 字段，在默认 absolute label path 下保持既有 sample keys。
- 提供无需训练模型即可运行的 distribution shift analysis，比较 source/target absolute beam 与 residual beam 分布距离。
- 增加单元测试和 smoke test，覆盖 split 防泄漏、target_labeled 5%、seed 可复现、残差 wrap-around 和诊断产物。

**Non-Goals:**

- 不新增 `GeoResidualBeamPredictor` 或训练一个新的 residual neural network。
- 不实现 AE/CL/MAE 预训练、feature cache 主线、target residual prior calibration 或 weather/reliability-aware fusion。
- 不改变旧 DeepSense6G、Raymobtime、MMW、HiST-Beam 配置的默认 label_space 和 split 行为。
- 不允许 target_test label、beam_power、path、radio/channel 字段参与 split 选择以外的训练、校准、阈值选择或 early stopping。

## Decisions

1. 使用独立 split builder 与 split artifact，而不是复用 `data.dataset.portion`。

   - 决策：新增 `split` 配置段，明确 source/target domain、`target_label_fraction=0.05`、selection strategy、seed、`allow_target_unlabeled` 和输出路径。split builder 写出 JSON/NPZ artifact，训练/诊断只消费 artifact。
   - 理由：`portion` 是 dataset 子采样，不表达 target_labeled/target_unlabeled/target_test 的防泄漏边界，也难以复现实验 split。
   - 替代方案：用现有 label budget 参数动态抽样 target support。该方案短期快，但每次运行可能隐式改变支持集，论文表格难复现。

2. domain key 由显式字段组合生成。

   - 决策：domain 定义支持 `scenario`、`weather`/`condition`、`scenario_weather`、`town_scenario_weather`，缺失字段时返回清晰错误或降级为配置声明的可用字段组合。
   - 理由：MMW 有 town/scenario/condition，DeepSense6G 和 Raymobtime 字段不完全一致；显式组合能避免把天气或场景混在 sample_id 字符串里。
   - 替代方案：只支持 MMW scenario-level split。该方案不能支撑附件中的多天气研究路线。

3. geometry-residual label 以通用工具为核心，MMW manifest geometry 作为数据源之一。

   - 决策：在轻量工具模块中实现 angle/beam/residual 运算；MMW 可复用 `preparation_geometry` 中的 relative azimuth/local pose，DeepSense6G/Raymobtime 可通过 GPS/BS position adapter 提供同等字段。
   - 理由：现有 `hist_beam_residuals` 是 history-anchor residual，以 last beam 为 anchor；本变更需要 geometry anchor，二者应共享 circular mapping 思路但不混淆字段语义。
   - 替代方案：直接扩展 history-anchor residual 字段。该方案容易把 `last_beam` anchor 和 `beam_geo` anchor 混在一起，造成评估解释错误。

4. clipped residual 是可选 label projection，不覆盖原始 residual。

   - 决策：始终保留 full circular residual 或 signed circular delta，`max_residual` 只用于生成 `residual_class`，clip/overflow 策略必须写入 metadata。
   - 理由：诊断需要真实 residual 分布；训练可选择较小的 residual class 空间，但不能丢失可审计标签来源。
   - 替代方案：直接把 residual 截断后作为唯一标签。该方案会掩盖 outlier 和 geometry 失败案例。

5. target_unlabeled 用 guard 保护监督字段。

   - 决策：split artifact 标记 target_labeled 与 target_unlabeled，dataset/runtime metadata 记录 labeled subset 状态；训练 loss 在 unlabeled target batch 访问 beam/residual/physical/path/radio supervision 必须失败。
   - 理由：5% target-shot 的核心是只用 target_labeled 监督，target_unlabeled 只能用于未来无监督/SSL，不应被基础实现意外泄漏。
   - 替代方案：dataset 直接删除 target_unlabeled 标签。该方案最安全，但会让同一底层 CSV/manifest 的评估和诊断复用变复杂；guard 更符合现有 sensitive field policy。

6. 分布诊断优先输出机器可读 JSON/CSV，图像为可选。

   - 决策：诊断命令必须写 `distribution_shift_metrics.json`、histogram CSV/JSON 和 summary；PNG/PDF 只有在可视化依赖可用且配置启用时生成。
   - 理由：测试和 summary 应依赖稳定结构化数据，避免 CI 或无显示环境受 matplotlib 影响。
   - 替代方案：只生成图片。该方案不利于自动汇总和回归测试。

## Risks / Trade-offs

- [Risk] 某些数据集缺少 BS/RSU pose 或 UE/CAV GPS，无法生成 geometry labels。Mitigation：`label_space.geometry.required` 控制失败或标记 unavailable；诊断输出 unavailable reason。
- [Risk] 5% target_labeled 在小 target domain 中样本过少，分层采样不稳定。Mitigation：记录最小样本数、实际 fraction、每 beam/sector 覆盖率；不足时输出 warning 并保持 seed 可复现。
- [Risk] circular residual 的符号约定与后续模型预期不一致。Mitigation：在 spec 和测试中固定 `beam_to_residual`/`residual_to_beam` 可逆关系，并在 metadata 写 `residual_convention`。
- [Risk] split artifact 与 CSV/manifest 版本不匹配。Mitigation：artifact 记录输入路径、样本数、sample id fingerprint、配置摘要和 seed；加载时不匹配则拒绝或要求 regenerate。
- [Risk] 新 target schema 触发既有 evaluator/batch preparation 的 key 假设。Mitigation：所有新字段只在 `label_space.type=geometry_residual` 或诊断命令中启用，默认 absolute label path 保持不变。

## Migration Plan

1. 新增轻量 split/geometry/diagnostics 模块和配置解析，不接入训练默认路径。
2. 接入 dataset runtime target provider，使显式 `label_space.type=geometry_residual` 时返回新增字段。
3. 接入 MMW split metadata 与 manifest geometry 字段，确保 target-shot artifact 可由现有 MMW prepared manifest 生成。
4. 新增诊断 CLI 或脚本，先支持从 split artifact 与 dataset 构建统计，不训练模型。
5. 新增测试并运行相关快速验证。
6. 回滚策略：停用新配置与入口即可恢复旧行为；新增 artifact 位于 outputs/cache 或用户指定目录，不进入源码。

## Open Questions

- DeepSense6G 和 Raymobtime 中 BS position 的 canonical 配置路径是否已经统一，还是需要为 geometry adapter 增加 dataset-specific 默认值。
- `angle_to_beam` 是否应默认使用真实 codebook 边界；若 codebook 不可用，本变更默认采用均匀 azimuth quantization，并在 metadata 中标记 `beam_geo_source=uniform_angle_quantization`。
- 对 clipped residual overflow 样本，训练阶段应映射到边界类还是 `ignore_index`，需要在后续 residual predictor change 中根据实验策略确定；本基础变更只要求 metadata 可区分。
