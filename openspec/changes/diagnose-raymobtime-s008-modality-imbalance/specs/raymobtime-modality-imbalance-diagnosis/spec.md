## ADDED Requirements

### Requirement: s008 诊断实验矩阵
系统 MUST 提供 Raymobtime s008 多任务模态失衡诊断实验矩阵，用于在固定数据契约下比较单任务、融合单任务、多任务任务组合、loss 权重和 checkpoint 选择的影响。该矩阵 MUST 优先使用 sensing-only `coord`、`image`、`lidar` 和 `coord+image+lidar` 条件，并 MUST 将包含 `ray` 的实验单独标记为 sensing+ray。

#### Scenario: 建立单任务多 seed 基线
- **WHEN** 用户执行 Raymobtime s008 模态失衡诊断矩阵
- **THEN** 系统 MUST 汇总 `coord`、`image`、`lidar` 和 `coord+image+lidar` 在 `current_beam_selection`、`current_los_classification` 和 `current_link_quality` 上的单任务结果
- **AND** 汇总 MUST 至少包含 seed42 已有结果，并为关键对照扩展到不少于三个训练 seed
- **AND** 汇总 MUST 记录每个 run 的 config 路径、resolved config、run 目录、enabled modalities、objective、seed、split metadata 和正式 validation metrics

#### Scenario: 执行多任务任务组合消融
- **WHEN** 用户执行 selection multitask 诊断消融
- **THEN** 系统 MUST 在相同 s008 cache、split 和 sensing-only CIL 输入下比较 `beam_only_multitask_model`、`beam+los`、`beam+link` 和 `beam+los+link`
- **AND** 每个组合 MUST 输出 beam Top-K/DBA、LOS F1/AUC、link MAE/R2、训练 loss 分量和 validation primary metric
- **AND** 报告 MUST 标记哪些任务在当前组合中被训练、哪些 metrics 只是未训练 head 的诊断输出

#### Scenario: 执行 loss 权重消融
- **WHEN** 用户执行 selection multitask loss 权重消融
- **THEN** 系统 MUST 至少比较 original 权重、equal 权重和 beam-heavy 权重
- **AND** original 权重 MUST 对应当前 s008 multitask 配置口径
- **AND** 报告 MUST 判断 beam 退化是否能被 loss 权重或 link loss 尺度调整恢复

#### Scenario: 执行 checkpoint 选择消融
- **WHEN** 用户分析任一 selection multitask run
- **THEN** 系统 MUST 汇总 best `val_selection_multitask_loss`、best `val_beam_top1` 和 best `val_link_mae` 对应 epoch 的指标
- **AND** 若可用 checkpoint 支持复评估，系统 MUST 对这些 epoch 的 checkpoint 使用同一 validation split 复评估
- **AND** 若 checkpoint 不可用，报告 MUST 基于 `train_log.json` 给出 epoch-level 指标并明确标记无法复评估

### Requirement: 模态失衡内部诊断产物
系统 MUST 为 Raymobtime s008 模态失衡诊断生成内部机制证据，包括按任务 gate、test-time modality drop、按任务/模态 gradient 或 contribution、按 LOS bucket 的 beam metrics。诊断产物 MUST 与性能矩阵使用同一 run 清单，并写入 ignored 输出目录。

#### Scenario: 输出 gate 诊断
- **WHEN** 诊断对象是 task-aware gated multitask selection run
- **THEN** 系统 MUST 输出按任务聚合的 gate 均值表
- **AND** 系统 MUST 输出按任务和 LOS bucket 聚合的 gate 均值表
- **AND** 表中 MUST 记录 run、task、modality、sample_count、gate_mean 和 gate_std

#### Scenario: 输出 test-time modality drop 诊断
- **WHEN** 诊断对象启用了两个或更多 sensing modality
- **THEN** 系统 MUST 分别报告 drop `coord`、drop `image` 和 drop `lidar` 后的 `beam_top1`、`beam_dba_current`、`los_f1` 和 `link_mae` 变化
- **AND** 若某个 modality 未启用，系统 MUST 在报告中标记该 drop 条件不可用，而不是输出误导性零变化

#### Scenario: 输出 gradient 或 contribution 诊断
- **WHEN** 诊断对象支持按任务反传或贡献估计
- **THEN** 系统 MUST 输出按 task 和 modality 聚合的 gradient norm 或 contribution 分数
- **AND** 报告 MUST 说明该分数的计算口径
- **AND** 若当前实现无法生成该诊断，系统 MUST 在诊断报告中列出缺失原因和所需后续实现

#### Scenario: 输出 LOS bucket 下的 beam 诊断
- **WHEN** Raymobtime s008 validation split 包含 LOS 标签
- **THEN** 系统 MUST 输出按 LOS bucket 分组的 beam Top-K 和 `beam_dba_current`
- **AND** 报告 MUST 比较 LOS/NLOS 条件下不同 modality 与 multitask run 的 beam 退化是否一致

### Requirement: 模态失衡判定标准
系统 MUST 基于预定义判定标准给出 s008 结论，结论 MUST 是 `confirmed_imbalance`、`likely_parameter_issue`、`inconclusive` 或 `diagnostics_blocked` 之一。判定 MUST 同时考虑外部性能、checkpoint 选择、loss 权重消融和内部诊断证据。

#### Scenario: 判定为 confirmed_imbalance
- **WHEN** 多 seed 下 multitask CIL 的 beam 指标稳定低于 beam 单任务最佳 sensing modality 或 beam-only CIL 对照
- **AND** best-by-beam checkpoint 视角仍不能恢复 beam 指标
- **AND** beam-heavy 或任务组合消融不能充分恢复 beam 指标
- **AND** gate/drop/gradient/LOS bucket 诊断至少两类支持任务或模态支配解释
- **THEN** s008 判定报告 MUST 将结论标记为 `confirmed_imbalance`

#### Scenario: 判定为 likely_parameter_issue
- **WHEN** beam-heavy、任务组合消融或 best-by-beam checkpoint 能将 beam 指标恢复到 beam 单任务 CIL 或最佳单模态附近
- **THEN** s008 判定报告 MUST 将结论标记为 `likely_parameter_issue`
- **AND** 报告 MUST 指出最可能的因素是 early stopping、loss 权重、loss 尺度或任务组合

#### Scenario: 判定为 inconclusive
- **WHEN** 多 seed 结果方向不一致或内部诊断证据不足以支持模态/任务支配解释
- **THEN** s008 判定报告 MUST 将结论标记为 `inconclusive`
- **AND** 报告 MUST 列出需要补跑的最小实验或诊断

#### Scenario: 判定为 diagnostics_blocked
- **WHEN** 关键 gate、drop、gradient 或 checkpoint 诊断无法生成
- **THEN** s008 判定报告 MUST 将结论标记为 `diagnostics_blocked`
- **AND** 报告 MUST 区分缺失原因是实验产物缺失、checkpoint 不可用、模型 diagnostics 未暴露还是分析工具能力不足

### Requirement: s009 外部验证门槛
系统 MUST 将 Raymobtime s009 作为第二阶段外部验证，而不是 s008 失衡确认的前置条件。只有当 s008 诊断报告达到 `confirmed_imbalance` 或高置信 `inconclusive` 且缺口不涉及代码/参数混杂时，系统 MAY 启动 s009 最小复刻矩阵。

#### Scenario: s008 未闭环时不得启动 s009 正式结论
- **WHEN** s008 判定为 `likely_parameter_issue`、`diagnostics_blocked` 或低置信 `inconclusive`
- **THEN** 报告 MUST 不使用 s009 结果作为 s008 模态失衡证据
- **AND** 后续任务 MUST 优先补齐 s008 参数、checkpoint 或诊断缺口

#### Scenario: s009 最小复刻矩阵
- **WHEN** s008 满足进入 s009 的门槛
- **THEN** s009 阶段 MUST 至少复刻 `lidar` 与 `coord+image+lidar` 的 beam/LOS/link 单任务、original multitask、beam-heavy multitask 和最可疑 task-combo
- **AND** s009 报告 MUST 明确记录数据契约、cache、split、label 对齐和可用 modality 与 s008 是否一致
- **AND** 若 s009 需要新增 dataset 或 preprocess 能力，系统 MUST 将其作为独立 OpenSpec change 处理

### Requirement: 诊断报告与产物边界
系统 MUST 生成 Raymobtime s008 模态失衡诊断报告和机器可读 summary，并 MUST 保证训练输出、日志、checkpoint、cache 和 TensorBoard 文件留在 ignored 输出目录。

#### Scenario: 生成诊断报告
- **WHEN** s008 诊断矩阵完成或因诊断缺口提前停止
- **THEN** 系统 MUST 在配置指定输出目录生成 summary JSON、run matrix CSV、metric comparison CSV、diagnostic tables 和 markdown 报告
- **AND** markdown 报告 MUST 包含实验矩阵、关键指标表、判定结论、证据链、反证检查和 s009 是否进入下一阶段的建议

#### Scenario: 保持本地产物边界
- **WHEN** 用户执行诊断实验或分析命令
- **THEN** 生成的训练日志、cache、checkpoint、TensorBoard 和临时报告 MUST 位于 ignored 输出目录
- **AND** 源码变更 MUST 只包含配置、脚本、测试或 OpenSpec artifacts，不得包含本地数据和新生成 checkpoint
