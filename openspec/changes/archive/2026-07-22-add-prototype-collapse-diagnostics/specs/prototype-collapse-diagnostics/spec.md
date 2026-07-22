## ADDED Requirements

### Requirement: 诊断必须冻结 checkpoint 与因果对照边界
系统 MUST 只分析 A0、A1、B2、C0、C7 的 validation-best checkpoint，并 MUST 在 manifest 中记录 config、selection、loss/router 开关、参数量和 prototype 属性。不存在同 split、未见 probe validation 且预算可比的 no-prototype checkpoint 时，系统 MUST 记录缺少公平因果对照且不得自动训练。

#### Scenario: 历史 NoBPA 已见 inner-validation
- **WHEN** 候选 no-prototype checkpoint 的训练集包含本轮 inner-validation 或训练预算/checkpoint policy 不可比
- **THEN** 系统 MUST 将其标记为非公平对照并排除主比较
- **AND** 不得因此启动补充训练

### Requirement: 固定样本与 corruption 必须 inner-only 且跨 checkpoint 一致
系统 MUST 只用 inner-train 拟合 probe/统计 train mean，只用 inner-validation 评测，并 MUST 以固定 seed、sample identity、sensor、corruption 和 severity 生成可配对 view。样本 manifest MUST 记录 split、weather、scene、beam、sensor、corruption、severity、missing 与 source frame identity。

#### Scenario: 重放五个 checkpoint
- **WHEN** 抽取器为不同 checkpoint 读取同一 sample manifest
- **THEN** clean/corrupt pair identity 与 manifest checksum MUST 完全一致
- **AND** outer test、channel、CSI、path、beam gain/power vector MUST 不被读取或缓存

### Requirement: 特征缓存必须表达真实层且允许 unavailable
缓存 MUST 记录 `F_enc`、`F_block`、`F_preproj`、`F_postproj`、`F_proto`、`Z_block`、`F_modality/Z_modality`、router、fused 与 availability 的真实来源、shape、dtype 和 alias。模型不存在某层时 MUST 写 `not_available`，不得伪造。

#### Scenario: 当前 64 维 prototype 路径
- **WHEN** 当前 encoder 内部末端 projection 后直接进入 prototype normalize/cosine
- **THEN** layer manifest MUST 记录 pre-projection hook、64 维 block、L2-normalized prototype feature 和 prototype bank 路径
- **AND** 默认关闭 intermediate return 时原 forward 与 state dict MUST 不变

### Requirement: D1-D6 必须联合评估 collapse、对齐、模态依赖与 Router 可观测性
系统 MUST 输出 prototype geometry/usage、逐层 spectrum/scatter/drift、beam/modality/corruption/severity/quality probe、cross-modal CKA/retrieval、single/LOMO/shuffle/replacement/gradient/unimodal evidence 和 R0/R1/R2 Router observability。probe 标准化与拟合 MUST 只使用 inner-train 且预算跨 checkpoint 固定。

#### Scenario: 低 effective rank 但 beam probe 保持
- **WHEN** projection 后 effective rank 下降但 beam Top-k/Within-3 保持或提高且 topology/margin 正常
- **THEN** 报告 MUST 将其视为良性 task compression 候选
- **AND** 不得仅凭 rank 或 CKA 宣称有害 collapse

### Requirement: Beam-conditional corruption collapse 必须按 BC1-BC7 判定
系统 MUST 按 checkpoint、modality、beam/sector、corruption、severity、weather 和 layer 输出 paired compression、severity centroid ratio、prototype residual probe、prototype sensitivity、hidden degradation score 与 prototype-gradient virtual step。只有 BC7 联合条件成立时才可判定有害 collapse。

#### Scenario: 良性 corruption invariance
- **WHEN** clean/severe feature 接近且 corrupted block task performance 未明显下降
- **THEN** 系统 MUST 记录良性 invariance 或无有害证据
- **AND** hidden degradation 不得被判为 collapse

### Requirement: 天气与模态结论必须保持分组公平性
主要 rank、probe、quality、shuffle/drop、weight 和 unimodal 指标 MUST 分 sunny/rainy/foggy 输出。不同天气样本没有严格 trajectory/frame/position identity 时 MUST 只分组，不得构造 paired weather comparison。

#### Scenario: 检查 LiDAR dominance
- **WHEN** 系统比较四模态 shuffle、missing、single-modality、gradient 与 unimodal evidence
- **THEN** 所有模态 MUST 使用相同 validation 样本和 shuffle seed
- **AND** LiDAR dominance MUST 由多项一致证据支持而非单个 missing 指标

### Requirement: 本地运行必须可恢复且不启动训练
runner MUST 在抽取前记录 GPU 状态，已存在且 checksum 匹配的 shard MUST 跳过；统计任务 MUST 独立记录 PID、日志与 return code，单任务失败不得阻止其他任务。runner MUST 不包含训练命令。

#### Scenario: 某个统计任务失败
- **WHEN** D1-D6 中任一任务非零退出
- **THEN** runner MUST 继续等待其他任务并运行聚合器
- **AND** summary MUST 将缺失证据标记 unavailable 而不是生成默认数值

### Requirement: 最终报告必须给出唯一主方向
`diagnostic_summary.md` MUST 回答附件列出的 prototype、feature、quality、CKA、modality、Router 与 weather 问题，并 MUST 从五个方向中选择唯一主方向，同时列出反证、限制和不确定性。所有结论 MUST 以 CSV 指标为主且保持 claim-ineligible。

#### Scenario: 聚合完成
- **WHEN** 可用诊断任务完成并生成 summary
- **THEN** 报告 MUST 分别判断 prototype collapse、feature collapse、quality erasure、LiDAR dominance 和 Router observability
- **AND** 不得修改正式 claim、canonical recipe 或自动实施建议方法
