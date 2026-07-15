## ADDED Requirements

### Requirement: T2 调参矩阵必须冻结架构与外层协议
系统 MUST 生成 `H0-base`、`H1-BPA+`、`H2-BPA-sharp`、`H3-mask-tail`、`H4-optimizer`、`H5-KL+`、`H6-teacher-low` 和 `H7-teacher-high` 八个 seed-1 development configs。所有 configs MUST 保持完整 T2 head、prototype package、circular geometry、四 sensing modalities、5-frame history、1-frame prediction、15 个 `weather/scenario` domains、outer train/test、40 epochs 和正式 evaluation masks 不变；每行只能包含 design 预注册且相对其 `matched_control` 允许的 override。

#### Scenario: 基线来源可审计
- **WHEN** launcher 生成 H0-H7
- **THEN** launcher MUST 读取 `outputs/mmw_all_weather_h5p1_seed1_v2/T2/seed1/resolved_config.yaml` 并记录路径与 SHA256
- **AND** launcher MUST 复用现有 MMW T2 builder，并在本地快照缺失、冻结字段不一致或 domain 数不为 15 时 fail closed

#### Scenario: 八行 resolved diff 受 allowlist 约束
- **WHEN** 任一 variant 完成 resolved config 生成
- **THEN** manifest MUST 记录 variant、matched control、canonical override、effective value 和完整 resolved diff
- **AND** resolved diff 出现未预注册 architecture、data、training、loss 或 evaluation 差异时 launcher MUST 拒绝该任务

#### Scenario: sharpness 行以 BPA strength 行为对照
- **WHEN** 生成 `H2-BPA-sharp`
- **THEN** 它相对 `H1-BPA+` MUST 只改变 prototype temperature 与 Gaussian sigma
- **AND** manifest MUST NOT 把 H2 相对 H0 的组合差异描述为单因素比较

### Requirement: 筛选必须使用独立开发验证集
系统 MUST 从每个 domain 的 outer train 侧按 MMW `group_safe_time_block` 规则生成固定 10% validation，并在同一 RSU time axis 上隔离不同 CAV 的共享 radar context；八个 variants MUST 共享相同 split artifacts。outer test MUST 保持不变，MUST NOT 作为 epoch validation、scheduler 选择、checkpoint selection 或 early stopping 数据。

#### Scenario: inner 训练与验证身份不相交
- **WHEN** inner split preflight 完成
- **THEN** 每个 15-domain inner train 与 validation MUST 在 stable sample、sequence group、history frame、target frame 和 referenced frame identity 上不相交
- **AND** 任一 inner role 为空、存在交集或缺少 split provenance 时 launcher MUST 不生成训练任务

#### Scenario: legacy outer test 的共享 RSU 上下文可见
- **WHEN** launcher 审计保留的 outer test
- **THEN** launcher MUST 记录跨 inner train/validation 与 outer test 的各模态资源重叠诊断
- **AND** launcher MUST NOT 将 legacy outer test 标记为全资源 strict-independent evidence

#### Scenario: 每 5 epoch 只做观测
- **WHEN** 一个筛选任务训练 40 epochs
- **THEN** 系统 MUST 使用独立 validation 至少记录 epoch 5、10、15、20、25、30、35、40 的观测
- **AND** config MUST 设置 `model_selection.enabled=false` 与 `use_early_stopping=false`
- **AND** validation 指标 MUST NOT 产生或选择 `best.pth`

#### Scenario: final test 只消费 last checkpoint
- **WHEN** 训练完成后执行筛选评估
- **THEN** evaluator MUST 只加载 fingerprint 匹配且完成 epoch 40 的 `last.pth`
- **AND** evaluator MUST 不回退到 `best.pth`、较早 checkpoint 或训练期 validation loader

### Requirement: 显存探测必须解析共同的 16 倍数批量大小且保持协议
系统 MUST 在每个目标 GPU 的全新子进程中，对 H0 的真实 MMW train batch 执行 AMP forward、loss、backward 和 optimizer step。候选 batch MUST 是运行前预注册的正 16 倍数；系统 MUST 选择所有目标 GPU 都成功且 peak reserved 不超过总显存 90% 的最大候选作为八行共同 batch。

#### Scenario: 所有 GPU 通过共同 batch
- **WHEN** 某候选在 GPU0-7 都完成真实 training step 且满足显存门槛
- **THEN** probe manifest MUST 将它记录为共同可用候选
- **AND** H0-H7 MUST 使用同一 `train_batch_size`

#### Scenario: probe 不联动修改 protocol
- **WHEN** probe 解析共同 batch
- **THEN** probe MUST NOT 修改或缩放 learning rate、optimizer、scheduler、gradient accumulation、epochs、split、loss、AMP、evaluation masks 或 checkpoint policy
- **AND** probe state、optimizer state 与 RNG state MUST NOT 传入正式训练任务

#### Scenario: 没有共同安全候选
- **WHEN** 每个预注册候选都在至少一个目标 GPU OOM、非零退出或超过显存门槛
- **THEN** launcher MUST fail closed 且不得启动任一正式任务
- **AND** launcher MUST NOT 为不同 GPU 或 variant 静默选择私有 batch

### Requirement: 生成配置和运行产物必须留在被忽略的输出根目录
系统 MUST 将 inner split CSV、generated YAML、manifest、probe report、logs、checkpoints、metrics、tables 和 figures 写入专用 ignored `outputs/` root。八个训练任务 MUST 使用互不覆盖的 run directory，并记录 variant、seed、GPU、common batch、baseline/config/split/mask fingerprints、checkpoint policy 和 development evidence flags。

#### Scenario: dry-run 生成本地产物
- **WHEN** 用户执行 launcher dry-run
- **THEN** 系统 MUST 只在 ignored output root 生成 split/config/manifest artifacts
- **AND** 系统 MUST 不在 `configs/`、`docs/`、`dataset/`、仓库根目录或 active BPA/CMA output root 写入 generated artifacts

#### Scenario: GPU0-7 并行映射
- **WHEN** 用户启动默认八行矩阵
- **THEN** launcher MUST 将 H0-H7 确定性映射到 GPU0-7，并保证每卡至多一个本矩阵训练进程
- **AND** run status MUST 记录启动、结束、退出码、完成 epoch 和 `last.pth` 完整性

### Requirement: 固定预算汇总必须使用预注册选择规则
系统 MUST 仅对通过完整性与身份校验的 epoch-40 `last.pth` 使用相同 MMW all-weather evaluator 和 mask artifacts。选择分数 MUST 为 `0.20*Clean + 0.20*mean(Drop1,Drop2,Drop3) + 0.25*temporal_AUC + 0.35*temporal_Drop80`；相对 H0 的 Clean、模态缺失均值或 temporal Drop80 下降超过绝对 `0.005` 的候选 MUST 被淘汰。

#### Scenario: 产生唯一 development candidate
- **WHEN** 所有八行完成且指标、sample identity、mask identity 与 fingerprints 一致
- **THEN** 系统 MUST 在通过保护门槛的行中按选择分数降序、variant id 升序确定唯一结果
- **AND** H0 排名第一时结果 MUST 记录为 `no_change`

#### Scenario: 汇总证据不完整
- **WHEN** 任一必要指标缺失或非有限、运行未满 40 epoch、checkpoint/config/split fingerprint 不匹配或 paired identity 不一致
- **THEN** summary MUST fail closed
- **AND** 系统 MUST NOT 补值、改变权重、选择较早 epoch 或排除失败样本后继续排名

### Requirement: 筛选证据必须与正式 BPA/CMA 变更隔离
本 change 的 configs、manifest、metrics 与 summary MUST 标记 `development_only=true`、`claim_eligible=false` 和 `screening_consumed_test=true`。系统 MUST NOT 修改或覆盖 active `validate-t2-mmw-bpa-cma-ablation` 的六方法定义、15 个 formal training jobs、配置、checkpoint、paired evaluation 或输出目录。

#### Scenario: development 结果不能直接升级 claim
- **WHEN** summary 选出 candidate 或 `no_change`
- **THEN** 系统 MUST 只记录 development screening 结论
- **AND** 系统 MUST NOT 将该结果写入 reviewed claim、论文主表或 formal multi-seed evidence

#### Scenario: 胜者需要后继 formal protocol
- **WHEN** 开发者希望把筛选胜者用于正式多 seed 比较
- **THEN** 必须先在本 change 之外冻结配置，并通过新的或显式扩展的 OpenSpec formal protocol 运行未参与调参的 evidence
- **AND** 本轮 test 指标 MUST NOT 被复用为论文 claim，也不得替换 active BPA/CMA change 中的 T2 control
