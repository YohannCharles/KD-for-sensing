## ADDED Requirements

### Requirement: T2 调参矩阵必须冻结架构与外层协议

系统 MUST 从 tracked T2 recipe 生成 `H0-base`、`H1-BPA+`、`H2-BPA-sharp`、`H3-mask-tail`、`H4-optimizer` 和 `H5-KL+` 六个 seed-1 development config。所有行 MUST 保持 T2 architecture、四模态、15-domain、40 epoch、outer train/test 和正式 evaluation masks 不变；每行只能包含相对其 `matched_control` 预注册的 override。

#### Scenario: 基线来源可审计

- **WHEN** launcher 生成 H0-H5
- **THEN** launcher MUST 读取 tracked `configs/mmw/t2.yaml` 及其 shared base 并记录 recipe SHA256
- **AND** recipe 缺失、冻结字段不一致或 domain 数不为 15 时 MUST fail closed

#### Scenario: 六行 resolved diff 受 allowlist 约束

- **WHEN** 任一 variant 完成 resolved config 生成
- **THEN** manifest MUST 记录 variant、matched control、canonical override、effective value 和 resolved diff
- **AND** 未预注册的 architecture、data、training、loss 或 evaluation 差异 MUST 被拒绝

### Requirement: 筛选必须使用独立开发验证集

系统 MUST 从每个 domain 的 outer train 侧按 current `group_safe_time_block` 规则生成固定 10% validation；六个 variants MUST 共享相同 split artifacts。outer test MUST NOT 用于 epoch validation、scheduler、checkpoint selection 或 early stopping。

#### Scenario: inner 训练与验证身份不相交

- **WHEN** inner split preflight 完成
- **THEN** inner train 与 validation MUST 在 sample、sequence group、history frame、target frame 和 referenced frame identity 上不相交
- **AND** 任一 role 为空、存在交集或缺少 provenance 时 launcher MUST 不生成训练任务

#### Scenario: 每五 epoch 只做观测

- **WHEN** 一个筛选任务训练 40 epochs
- **THEN** 系统 MUST 使用独立 validation 至少记录每五 epoch 观测
- **AND** config MUST 禁用 model selection 和 early stopping，且 validation MUST NOT 选择 `best.pth`

### Requirement: 显存探测必须解析共同的 16 倍数批量大小且保持协议

系统 MUST 在每个请求 GPU 的新子进程中，对 H0 的真实 MMW train batch 执行 AMP forward、loss、backward 和 optimizer step。候选 batch MUST 是预注册的正 16 倍数；系统 MUST 选择所有请求 GPU 都成功且 peak reserved 不超过总显存 90% 的最大候选作为六行共同 batch。

#### Scenario: 所有目标 GPU 通过共同 batch

- **WHEN** 某候选在所有请求 GPU 完成真实 training step 且满足显存门槛
- **THEN** probe manifest MUST 记录共同可用候选
- **AND** H0-H5 MUST 使用同一 `train_batch_size`

#### Scenario: 没有共同安全候选

- **WHEN** 每个预注册候选都在至少一个目标 GPU OOM、非零退出或超过显存门槛
- **THEN** launcher MUST fail closed 且不得启动正式任务
- **AND** launcher MUST NOT 为不同 GPU 或 variant 选择私有 batch

### Requirement: 生成配置和运行产物必须留在被忽略的输出根目录

系统 MUST 将 inner split CSV、generated YAML、manifest、probe report、logs、checkpoints、metrics、tables 和 figures 写入专用 ignored `outputs/` root。六个训练任务 MUST 使用互不覆盖的 run directory，并记录 variant、seed、GPU、common batch、recipe/config/split/mask fingerprints、checkpoint policy 和 development evidence flags。

#### Scenario: dry-run 生成本地产物

- **WHEN** 用户执行 launcher dry-run
- **THEN** 系统 MUST 只在 ignored output root 生成 split/config/manifest artifacts
- **AND** 系统 MUST 不在 tracked config、docs、dataset、仓库根目录或 active BPA/CMA output root 写入 generated artifacts

### Requirement: 固定预算汇总必须使用预注册选择规则

系统 MUST 仅对通过完整性与身份校验的 epoch-40 `last.pth` 使用相同 MMW all-weather evaluator 和 mask artifacts。选择分数 MUST 为 `0.20*Clean + 0.20*mean(Drop1,Drop2,Drop3) + 0.25*temporal_AUC + 0.35*temporal_Drop80`；相对 H0 的 Clean、模态缺失均值或 temporal Drop80 下降超过绝对 `0.005` 的候选 MUST 被淘汰。

#### Scenario: 产生唯一 development candidate

- **WHEN** 所有六行完成且指标、sample identity、mask identity 与 fingerprints 一致
- **THEN** 系统 MUST 在通过保护门槛的行中按选择分数降序、variant id 升序确定唯一结果
- **AND** H0 排名第一时结果 MUST 记录为 `no_change`

### Requirement: 筛选证据必须与正式 BPA/CMA 变更隔离

本 change 的 configs、manifest、metrics 与 summary MUST 标记 `development_only=true`、`claim_eligible=false` 和 `screening_consumed_test=true`。系统 MUST NOT 修改或覆盖 active BPA/CMA formal protocol。

#### Scenario: development 结果不能直接升级 claim

- **WHEN** summary 选出 candidate 或 `no_change`
- **THEN** 系统 MUST 只记录 development screening 结论
- **AND** 系统 MUST NOT 将该结果写入 reviewed claim、论文主表或 formal multi-seed evidence
