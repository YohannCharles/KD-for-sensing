## ADDED Requirements

### Requirement: strict evidence provenance 必须从训练传播到评估和汇总
带有 `mmw_twc_evidence` 的训练 config、checkpoint、fixed-mask row 和 summary MUST 记录 evidence protocol id、split manifest SHA256、split role、training mask seed algorithm、evaluation mask cache checksum、seed、recipe fingerprint、training/router profile 和 topology descriptor。任一 identity 缺失或不一致 MUST fail closed。

#### Scenario: 评估 strict checkpoint
- **WHEN** evaluator 加载 strict evidence checkpoint
- **THEN** 它 MUST 验证 checkpoint 与 generated config 的 strict evidence identity
- **AND** 所有输出 row MUST 继承相同 identity

#### Scenario: 训练 mask 可重现
- **WHEN** 同一 strict config 在相同 seed 下从头训练
- **THEN** training temporal missing MUST 使用可记录的 deterministic seed algorithm
- **AND** provenance MUST 区分 training missing seed 与 immutable evaluation mask cache seed

### Requirement: evaluation label 必须来自 current runtime batch contract

共享 MMW evaluation helper MUST 从 `TaskForwardResult.batch` 通过 `prepare_task_labels` 取得目标标签，而不得访问未声明的 `TaskForwardResult` 字段。prediction slot 与 label slot MUST 使用相同的 `num_pred` 语义。

#### Scenario: TaskForwardResult 不携带 labels
- **WHEN** runtime forward 返回只包含 `batch`、`model_output` 和 `logits` 的 `TaskForwardResult`
- **THEN** fixed-mask evaluation MUST 仍能计算与最后 prediction slot 对齐的 beam metrics
- **AND** 不得因访问 `step.labels` 而失败

### Requirement: 六方法训练必须共享唯一外部缺失 mask
训练 runtime MUST 以互斥类别生成 Clean 20%、Drop1/Drop2/Drop3 各 10%、TokenDrop20/40/60/80/90 各 10% 的可复现外部 mask。T2 和 S1 MUST 直接消费外部 availability，不得再执行独立 `p_missing` 采样；MaskTrain-CLS、AMBER-Full、RMBP-MM 与 AMR-Net-4M-Adapted MUST 接收相同的外部 modality-temporal mask。

#### Scenario: 同 seed 六方法公平输入
- **WHEN** 六方法以相同 seed、epoch、step 和 batch inventory 训练
- **THEN** 每个 sample 的外部 mask 与 condition id MUST 完全相同
- **AND** T2/S1 extension MUST 不产生第二套随机输入 mask

### Requirement: current T2/S1 必须使用普通 Beam CE
T2 和 S1 MUST 使用相同的普通 Beam CE。系统 MUST 不保留 Pattern-weighted CE 配置、运行分支、launcher variant 或 summary 完整性要求。

#### Scenario: 构建 current T2/S1
- **WHEN** current recipe 或 strict launcher 构建 T2/S1
- **THEN** loss MUST 直接计算普通 Beam CE
- **AND** retired Pattern variant MUST 返回普通 unknown-name 错误而不是兼容映射

### Requirement: WholeOnly 必须使用独立的平衡 whole-modality panel
WholeOnly training MUST 使用 `mmw_fair_whole_modality_v1` 480-entry deterministic panel，其中 Clean、Drop1、Drop2、Drop3 各 120 项，且每个 drop count 内的模态组合精确等频；token20/40/60/80/90 的 condition count MUST 显式为 0。该 panel MUST 不包含 modality-frame、frame-level 或 block token missing，并 MUST 记录独立的 schedule id、panel size、condition counts、seed algorithm 与 panel checksum。

#### Scenario: 构建 WholeOnly matched control
- **WHEN** strict launcher 构建 T2-WholeOnly
- **THEN** active mask scheduler MUST 实际消费 whole-only panel，而不得只写入被忽略的 rate/type 字段
- **AND** 同 seed/epoch 的 panel MUST 可复现且与 T2 的 600-entry structured panel identity 不同

### Requirement: Router utility screen 不得修改源数据或提前提交后续 seed
`mmw_router_utility_screen_v1` MUST 只对 inner-train batch 做 deterministic online corruption，并 MUST 从原始只读 future beam-power path 加载 auxiliary target。训练 corruption seed MUST 与 fixed Oracle Gap evaluation seed 不同；launcher MUST 只生成 seed1 八候选，用户确认前不得生成或提交 seed2--5。

#### Scenario: 八卡 seed1 开发筛选
- **WHEN** 用户授权 GPU0--7 运行 utility screen
- **THEN** 八候选 MUST 各绑定一个物理 GPU 并共享 H4、RouterNoPattern、40 epoch、batch64、missing schedule 和 inner split
- **AND** config/manifest/checkpoint MUST 记录 full-power privileged supervision、paired corruption 参数、loss weights 与 claim-ineligible 身份
- **AND** 训练完成后的 corruption evaluation MUST 使用独立 seed 与完整 15-domain sample identity

### Requirement: expected-utility 修复筛选必须在提交前 fail closed
`mmw_router_expected_utility_screen_v3` launcher MUST 只生成 seed1 八候选，并 MUST 在启动 GPU0--7 子进程前验证固定 inner traces 的 12 个 sensor-severity cells、sample identity、expected-utility active ratio、成熟 target entropy gate coverage 和 focused gradient smoke。preflight 产物 MUST 标记 `claim_eligible=false`，且不得作为 outer paper claim。

#### Scenario: 任一传感器没有连续质量信号
- **WHEN** Image、Radar、GPS 或 LiDAR 在全部三个 severity 下都没有超过 gate epsilon 的样本
- **THEN** launcher MUST 在创建任何训练进程前失败
- **AND** manifest MUST 不得把未启动候选标记为 running 或 complete

#### Scenario: 八卡训练后的 Oracle Gap 评估
- **WHEN** 八个 seed1 checkpoint 完整且通过身份校验
- **THEN** 每候选 MUST 使用独立 evaluation corruption seed 完成 15-domain、13-condition 固定评估
- **AND** condition evaluator MUST 隔离资源生命周期，单个 Python 进程不得因累计 dataloader/file descriptor 泄漏跨越全部 condition
