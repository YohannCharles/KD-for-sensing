## ADDED Requirements

### Requirement: 决策适配实验保持最小公共表面和本地产物边界
Prototype-conditioned decision Adapter MUST 作为受协议约束的本地实验工作流和分析工具实现，不得新增 public CLI、canonical MMW recipe、历史 route 或兼容入口。checkpoint、mask schedule、逐样本 NPZ、日志、统计报告和运行 manifest MUST 写入 `outputs/prototype_decision_adapter/`，不得被源码或 current spec 读取为运行依赖。

#### Scenario: 无本地实验产物的源码检查
- **WHEN** 仓库在没有 `outputs/prototype_decision_adapter/` 的环境中加载 canonical recipe、CLI help 或执行架构边界测试
- **THEN** 当前 U0、AMBER-Full、RMBP-MM 与 DeepSense6G 的现有公共工作流 MUST 继续可用

### Requirement: Full-pool capacity 产物和编排保持本地边界
Full-pool protocol、增强 cache、动态生成配置、checkpoint、逐样本 logits、运行日志与报告 MUST 写入 `outputs/full_pool_capacity/`。`scripts/run_full_pool_capacity_gpu4_7.sh` MUST 只是受协议约束的本地编排脚本，不得新增 public CLI、canonical recipe、系统启动项或凭证文件修改。

#### Scenario: 长时间 GPU1--7 运行
- **WHEN** 编排器启动或等待 Stage 1/2 子任务
- **THEN** 它 MUST 每 600 秒向 runtime 产物记录简洁状态，并只在启动、退出、错误或完成时额外检查
- **AND** 它不得终止无关 GPU 进程或通过系统配置实现自启动

### Requirement: Full-pool 缓存加速不恢复历史运行入口
帧缓存 reader、GPS cache manifest 与可选 LMDB benchmark MUST 保持在当前 `src/kd_sensing` 数据边界和 `outputs/full_pool_capacity/` 本地产物边界内。系统 MUST 不恢复旧 sample-LMDB public CLI、canonical recipe 或兼容聚合层，也不得删除、覆盖现有 `outputs/cache/MMW`。

#### Scenario: 旧 LMDB 存在
- **WHEN** Full-pool workflow 发现历史 split-level sample LMDB
- **THEN** 它 MUST 不将其作为当前四模态 Full-pool 训练输入
- **AND** 仅可在预注册吞吐门槛失败后生成新的唯一帧分片 LMDB

### Requirement: ADBA-surrogate 产物独立于原 A 组
B1/B4/B6/B7 的 checkpoint、逐样本预测、日志、运行 manifest 与独立重算 MUST 写入 `outputs/full_pool_capacity/adba_surrogate/`，不得覆盖 `stage2/a0`--`stage2/a7` 或新增 public CLI、canonical recipe 和系统启动项。

#### Scenario: B 组重复启动
- **WHEN** 目标 B run 目录已存在
- **THEN** 工作流 MUST fail closed，不得隐式覆盖或将不完整目录当作完成结果

### Requirement: Mask-bias novelty triage 使用独立本地产物根
Global/Lookup、weight-space probe、unseen fold schedule、MLP/Factorized checkpoint、逐样本预测与独立重算 MUST 写入 `outputs/full_pool_capacity/mask_bias_ablation/`，不得覆盖 A 组、B 组、U0 或 protocol 产物，也不得新增 public CLI 或 canonical recipe。

#### Scenario: 条件阶段重复启动
- **WHEN** `mask_bias_ablation` 目标目录已存在
- **THEN** 编排器 MUST fail closed，不得隐式复用不完整阶段或覆盖现有结果

### Requirement: 环形传输实验使用独立本地产物根
Circular Transport、all-seen Factorized Bias checkpoint、逐样本预测、局部核审计、运行 manifest 与独立重算 MUST 写入 `outputs/full_pool_capacity/circular_transport/`，不得覆盖既有 U0、A/B 组或 mask-bias triage 产物，也不得新增 public CLI 或 canonical recipe。

#### Scenario: 环形传输重复启动
- **WHEN** `circular_transport` 目标目录已存在
- **THEN** 编排器 MUST fail closed，不得覆盖、续写或隐式复用该目录

### Requirement: BT-SCL 产物与启动表面保持本地
Full-Pool BT-SCL 的 protocol copies、topology audit、initialization、schedule、normalization、checkpoint、诊断、GPU orchestration 和最终报告 MUST 写入 `outputs/full_pool_bt_scl/`。`tools/run_full_pool_bt_scl.py` 与 `scripts/run_full_pool_bt_scl_gpu0_5.sh` 仅是本地实验工具，不得新增 public CLI、canonical recipe 或系统启动项。

#### Scenario: 无 BT-SCL 本地产物
- **WHEN** canonical recipe、public CLI help 或架构边界测试在没有 `outputs/full_pool_bt_scl/` 的环境运行
- **THEN** current U0、保留 baseline 与 DeepSense6G 工作流 MUST 不依赖 BT-SCL 产物

### Requirement: R6 结果不覆盖 R0--R5 原始运行
R6 checkpoint、训练曲线、15-pattern 指标和多半径机制诊断 MUST 写入 `outputs/full_pool_bt_scl/r6_topological_stochastic_dominance/`；R0 机制补算 MUST 写入独立只读派生产物，不得覆盖 R0--R5 checkpoint、训练曲线或原始 metrics。

#### Scenario: R6 重复启动
- **WHEN** R6 正式运行目录已包含结果产物
- **THEN** runner MUST fail closed，不得隐式覆盖或续写

### Requirement: R6 stable follow-up 使用独立产物根
post-hoc stable R0/R3/R6 的配置、checkpoint、训练曲线、返回码、指标和机制诊断 MUST 写入 `outputs/full_pool_bt_scl_stable_r6/`。它 MAY 只读复用 `outputs/full_pool_bt_scl/` 的初始化、train-only normalization 和 schedule，但 MUST 校验对应哈希且不得改写原产物。

#### Scenario: stable follow-up 重复启动
- **WHEN** 任一 stable 正式 run 目录已包含结果
- **THEN** runner MUST fail closed，不得覆盖或将其混入原 R0--R6 汇总

### Requirement: Candidate12 使用独立本地产物和脚本
Candidate12 的协议审计、warm-up、assignment、A0--A5 checkpoint、诊断和报告 MUST 写入 `outputs/full_pool_candidate12_search/`。`tools/run_full_pool_candidate12.py`、`scripts/run_full_pool_candidate12_gpu0_5.sh` 与 `scripts/monitor_full_pool_candidate12.sh` MUST 仅为本地实验工具，不得新增 public CLI、canonical recipe、系统启动项或覆盖既有 U0/Adapter/BT-SCL 产物。

#### Scenario: Candidate12 重复启动或单任务失败
- **WHEN** 正式目标目录已有结果，或任一并行任务失败
- **THEN** runner MUST 拒绝覆盖，其他任务 MAY 继续且必须各自记录 exit code
- **AND** 不得自动改超参数、重启、启动 multi-seed、outer test 或下一轮实验

### Requirement: BTMA 收尾写入独立只读派生产物
BTMA 收尾的逐样本预测、block bootstrap、score correlation 与报告 MUST 写入 `outputs/btma_posthoc_closure/`。`tools/analyze_btma_closure.py` MUST 仅为本地只读分析工具，MUST NOT 新增 public CLI 或 canonical recipe，MUST NOT 改写 `outputs/full_pool_btma_ablation/` 下的任何既有产物。

#### Scenario: 收尾重复启动
- **WHEN** 收尾目标目录已包含结果产物
- **THEN** 工具 MUST fail closed，不得隐式覆盖或续写
- **AND** MUST NOT 因收尾结果自动启动 multi-seed、outer test 或下一轮实验

### Requirement: Router 筛选使用独立本地产物和脚本
Router 可观测性筛选的表征缓存、腐蚀条件抽取、Q0--Q3 各 seed 的 checkpoint 与指标、推理期消融、success gates 与报告 MUST 写入 `outputs/router_observability/`。相关运行器与启动脚本 MUST 仅为本地实验工具，MUST NOT 新增 public CLI、canonical recipe 或系统启动项，MUST NOT 改写 `outputs/full_pool_capacity/` 下的冻结 U0 产物。

#### Scenario: Router 筛选重复启动或单路线失败
- **WHEN** 任一 setting/arm/seed 目录已有结果，或任一路线失败
- **THEN** runner MUST 拒绝覆盖，其他路线 MAY 继续且必须各自记录 exit code
- **AND** 不得自动改超参数、重启、启动多 seed 骨干、outer test 或下一轮实验

#### Scenario: 无 Router 筛选本地产物
- **WHEN** canonical recipe、public CLI help 或架构边界测试在没有 `outputs/router_observability/` 的环境运行
- **THEN** current U0、保留 baseline 与 DeepSense6G 工作流 MUST 不依赖该筛选产物
### Requirement: sparse-pilot 产物保持本地且不扩展公共入口
Probe codebook、noiseless candidate cache、prototype lookup、resolved config、diagnostics、ablation/SNR/budget summary、checkpoint 与日志 MUST 写入 `outputs/sparse_pilot_transition/` 或 `outputs/cache/`。相关 runner/build/eval 工具 MUST 仅为本地实验入口，不得新增 public CLI、canonical MMW recipe、系统启动项或源码对本地产物的导入依赖。

#### Scenario: 无 sparse-pilot 本地产物
- **WHEN** canonical recipe、CLI help、model construction 或架构边界测试在没有 `outputs/sparse_pilot_transition/` 的环境运行
- **THEN** 当前 U0、AMBER-Full、RMBP-MM 与 DeepSense6G MUST 继续可解析和运行

### Requirement: dense-to-sparse 结果不覆盖首轮 sparse-pilot 诊断
Dense-to-sparse 的 codebook、cache manifest、lookup、budget curriculum、SNR/mask 汇总与报告 MUST 写入独立的 `outputs/sparse_pilot_transition_dense_to_sparse/` 和对应 `outputs/cache/` 根，不得覆盖首轮 M4/K8 负结果，也不得新增 public CLI 或 canonical recipe。

#### Scenario: 运行 dense-to-sparse 诊断
- **WHEN** 本地 runner 使用 dense-to-sparse config
- **THEN** resolved config 与每阶段实际 M/Kp/sounding/RE MUST 写入新产物根
- **AND** 原 `outputs/sparse_pilot_transition/` MUST 保持不变

### Requirement: sparse-pilot scale-up 保持独立可恢复证据
Scale-up 四路的 resolved config、均衡索引摘要、逐 epoch history、轻量 trainable-only checkpoint、返回码和汇总 MUST 写入独立的 `outputs/sparse_pilot_scale_up_2k_40e/`。checkpoint MUST NOT 复制冻结 U0 参数；runner 不得扩展 public CLI 或 canonical recipe。

#### Scenario: scale-up 单路失败
- **WHEN** 任一 GPU0--3 arm 失败或既有 GPU 负载变化
- **THEN** 其他 arm MAY 继续并各自记录返回码
- **AND** workflow MUST NOT 终止既有进程、自动降低 batch、改超参数、覆盖目标目录或隐式重跑

### Requirement: missing-fallback 结果与 Full scale-up 隔离
严重缺失兜底四路的 mask schedule audit、逐 epoch severe-validation history、trainable-only checkpoint、15-mask CSI on/off summary 与报告 MUST 写入独立的 `outputs/sparse_pilot_missing_fallback_2k_40e/`，不得覆盖既有 Full scale-up 证据或扩展公共 CLI。

#### Scenario: 兜底实验完成
- **WHEN** GPU0--3 四路 missing-fallback 训练完成
- **THEN** 每路 MUST 保存相同 train/validation 子集哈希、训练 mask cardinality 计数、返回码和 outer-test 未访问标志
- **AND** 汇总 MUST 先报告 Single Macro/Worst 与 All-14 Macro 的 CSI-on/off 差值，再报告 Full 不伤害结果

### Requirement: missing-fallback 中间预算独立保存
D16x16、S8x16、S16x8 与 S8x8 的 config、history、checkpoint、15-mask summary、返回码和汇总 MUST 写入 `outputs/sparse_pilot_missing_fallback_midbudgets_2k_40e/`，不得覆盖首轮 missing-fallback 四路。

#### Scenario: 中间预算单路失败
- **WHEN** 任一 GPU0--3 中间预算 arm 失败
- **THEN** 其他 arm MAY 继续并独立记录返回码
- **AND** workflow MUST NOT 自动调整 M、Kp、batch、epoch、loss 或重新启动失败任务

### Requirement: sparse-pilot recovery 使用独立本地产物根
CSI 信息诊断、resolved configs、逐 epoch 日志、完整恢复 checkpoint、采样统计和阶段报告 MUST 写入 `outputs/sparse_pilot_recovery/`。相关工具 MUST 保持本地实验入口，不得新增 public CLI、canonical recipe、系统启动项或源码对该目录的依赖。

#### Scenario: 并行运行 I1--I5
- **WHEN** GPU0--3 并行执行三 seed 信息诊断
- **THEN** 每个 seed/task MUST 写入独立日志、checkpoint、结果与返回码，汇总器只读这些完成产物
- **AND** 单任务失败不得终止无关进程、自动改超参数、访问 outer test 或直接启动 Stage A1

### Requirement: trajectory recovery 三轮不得覆盖旧协议证据

纠错后的最大母 cache、冻结 M4 表征、三轮 config/log/checkpoint/result/analysis MUST 写入 `outputs/sparse_pilot_recovery/trajectory_v1/` 与对应 `outputs/cache/` 子根。旧 Full-pool recovery 产物 MUST 保持不变；每个并行任务 MUST 独立记录返回码，长任务状态轮询间隔 MUST 为 600 秒。

#### Scenario: 三轮顺序执行

- **WHEN** Round 1 或 Round 2 尚未完成分析
- **THEN** workflow MUST 不启动下一轮
- **AND** 任一 GPU 满载时 MAY 等待空闲，但 MUST 不终止、迁移或抢占无关进程
