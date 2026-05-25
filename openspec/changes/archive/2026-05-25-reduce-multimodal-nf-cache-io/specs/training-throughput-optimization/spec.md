## ADDED Requirements

### Requirement: Multimodal-NF cache IO profiling
训练吞吐 profile MUST 对 Multimodal-NF image/LiDAR 派生 cache 的初始化、校验、打开和读取耗时提供结构化诊断。profile 输出 MUST 能区分 cache validation、DataLoader wait、CPU 到 GPU transfer、forward/backward 和 optimizer step，避免把 IO 等待误判为 CUDA 或模型结构问题。

#### Scenario: 输出 cache 校验耗时
- **WHEN** 用户对 Multimodal-NF image/LiDAR/fusion 配置运行训练吞吐 profile
- **THEN** profile 输出 MUST 记录每个启用派生 cache 模态的 validation mode、validation duration、是否执行 source fingerprint scan、cache policy 和 source kind
- **AND** profile 输出 MUST 记录 dataset 构建或 cache plan 解析耗时

#### Scenario: 输出 cache open/read 耗时
- **WHEN** profile 读取 Multimodal-NF image/LiDAR 样本
- **THEN** profile 输出 MUST 记录 cache open 次数、cache read 次数、cache read 平均值、P95、最大值和按模态汇总的 `__getitem__` 耗时
- **AND** 输出 MUST 包含 cache path 数量、cache 总字节数、storage kind、layout 和推荐访问模式

#### Scenario: 输出 IO-risk 判定
- **WHEN** DataLoader wait 的 P95 明显高于模型 step 或 cache read 尾延迟明显高于均值
- **THEN** profile 输出 MUST 用结构化字段标记 IO-risk，例如随机读风险、loader wait dominates step、cache validation scan detected 或 mmap page fault risk
- **AND** profile 输出 MUST 保留原有 samples/s、CUDA peak memory、worker、prefetch、pin memory、AMP 和 progress 字段

### Requirement: Multimodal-NF train 子采样局部性控制
训练流程 MUST 支持或推荐 Multimodal-NF image/LiDAR 大 cache 场景下的局部性优先 train 子采样顺序。该能力 MUST 保持样本选择可复现，并 MUST 在 run metadata 中记录完整 train 样本数、有效样本数、seed、轮换策略和输出顺序策略。

#### Scenario: 随机选择后按局部性排序
- **WHEN** train epoch 子采样启用且用户选择局部性优先顺序
- **THEN** 每个 epoch MUST 先基于 seed 和 epoch 生成可复现的无放回样本子集
- **AND** 输出给 DataLoader 的样本顺序 MUST 按 dataset index、source key、block key 或等价局部性键排序或分块
- **AND** 有效样本集合 MUST 与同 seed 下的随机选择语义一致，除非配置显式选择固定顺序调试模式

#### Scenario: 旧随机顺序兼容
- **WHEN** 用户未配置局部性优先顺序且保持现有 `shuffle=true`
- **THEN** train 子采样 MUST 保持现有随机顺序语义
- **AND** 现有训练配置 MUST 不需要修改即可运行

#### Scenario: 运行产物记录局部性策略
- **WHEN** 局部性优先顺序被启用或由推荐器输出
- **THEN** 最终配置、运行元数据或 epoch 日志 MUST 记录排序策略、是否 block shuffle、block size 或等价参数
- **AND** 用户 MUST 能区分完整随机 batch 顺序和 IO-friendly batch 顺序

### Requirement: Multimodal-NF 并行训练 IO 推荐
并行训练推荐器 MUST 对 Multimodal-NF image/LiDAR/fusion 运行输出 IO-aware 覆盖参数和说明。推荐结果 MUST 不直接修改用户配置文件，MUST 明确区分后台并行训练建议与单实验默认值。

#### Scenario: image/LiDAR/fusion 后台并行推荐
- **WHEN** 用户为 Multimodal-NF image/LiDAR/fusion 配置请求并行训练推荐
- **THEN** 推荐器 MUST 输出关闭 batch progress、合理 worker/prefetch、cache policy、cache validation mode 和局部性优先 train 子采样顺序建议
- **AND** CUDA image/fusion 训练推荐 MUST 包含 AMP 覆盖或明确说明 AMP 适用条件
- **AND** 推荐说明 MUST 提醒用户避免把 `read_only` cache 等同于必然高速路径

#### Scenario: cache warm 后避免重复强校验
- **WHEN** 推荐器判断 Multimodal-NF image/LiDAR 派生 cache 已预热且 sidecar 可轻量校验
- **THEN** 推荐器 MUST 建议 warm cache 训练使用轻量运行时校验
- **AND** 推荐器 MUST 不建议每个并行训练进程重复执行原始 HDF5 全量 fingerprint scan

#### Scenario: GPU 分配和重 IO run 提示
- **WHEN** 用户请求多个 Multimodal-NF image/LiDAR/fusion 后台训练并行运行
- **THEN** 推荐输出 MUST 包含重 IO run 的 GPU 分配提示，建议优先跨 GPU 均匀分配并避免在同一 GPU 上叠加多个重 IO run
- **AND** 当并行数超过 GPU 数或 cache IO-risk 较高时，推荐输出 MUST 提示降低并行度、启用局部性顺序或先运行小样本 profile

#### Scenario: profile 驱动推荐
- **WHEN** 用户提供或生成了 Multimodal-NF profile 输出
- **THEN** 推荐器 MUST 使用 profile 中的 DataLoader wait、cache read P95、IO-risk 和 GPU step 字段调整 worker、局部性顺序和 AMP 建议
- **AND** 如果没有 profile，推荐器 MUST 给出保守默认和运行 profile 的命令提示
