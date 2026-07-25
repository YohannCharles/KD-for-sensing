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
