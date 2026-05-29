## ADDED Requirements

### Requirement: HiST-Beam source prototype 按需生成
HiST-Beam LOSO executor MUST 根据 variant 和配置决定是否生成 source prototype。只有后续 stage 需要 prototype 的 variant 或用户显式要求保存 prototype 时，source training 才应生成 prototype artifact。

#### Scenario: source-only baseline 跳过 prototype
- **WHEN** LOSO run 的 source variant 为 `v0_flat`、`v1_hierarchical`、`v2_shared_private` 或其它不需要 target prototype alignment 的 source-only baseline
- **THEN** source training 默认 MUST 跳过 source prototype 生成
- **AND** run metadata MUST 记录 prototype status 为 `skipped` 及跳过原因

#### Scenario: prototype variant 按需生成或复用
- **WHEN** 后续 `v5_adapter_proto`、`v6_radio_proto` 或 `adapter_radio_proto` stage 需要 source prototype
- **THEN** executor MUST 生成或复用与 fold、source scenes、variant、seed 和 prototype type 匹配的 source prototype artifact
- **AND** 若 artifact 不可用，target adaptation MUST 给出清晰失败或 no-op 诊断

### Requirement: Source prototype 进度与耗时诊断
Source prototype 生成 MUST 提供 stage progress 和耗时诊断，避免 image-heavy source split 二次扫描时表现为无进度卡死。

#### Scenario: prototype pass 写出 progress
- **WHEN** executor 正在生成 source prototype
- **THEN** stage progress MUST 周期性记录 processed batches、total batches 或可用近似进度
- **AND** progress MUST 标明当前 phase 为 `source_prototype`

#### Scenario: prototype metrics 记录额外扫数成本
- **WHEN** source prototype 生成完成
- **THEN** metrics MUST 记录 prototype generation duration、processed sample count、processed batch count 和 prototype coverage
- **AND** LOSO summary MUST 能区分 source training time 和 prototype generation time

### Requirement: MMW HiST-Beam LOSO stage 内存边界
HiST-Beam MMW LOSO 执行器 MUST 在每个 stage 结束后关闭不再需要的 DataLoader worker，并释放 stage-local dataset/loader 引用，使后续 stage 或 run 不继承 image-heavy worker 内存。

#### Scenario: source stage 结束释放 loader
- **WHEN** `source_train` stage 完成、失败或被中断
- **THEN** executor MUST 关闭 source DataLoader worker
- **AND** stage metadata MUST 不保留不可序列化的大 dataset 或 loader 对象

#### Scenario: run summary 记录吞吐配置
- **WHEN** MMW HiST-Beam run 完成或 partial failure
- **THEN** run metadata 或 summary MUST 记录 batch size、num_workers、persistent_workers、prefetch_factor、enabled modalities、seq_len、image cache policy 和 prototype strategy
- **AND** 这些字段 MUST 足以解释 GPU 低利用率和 CPU 内存压力
