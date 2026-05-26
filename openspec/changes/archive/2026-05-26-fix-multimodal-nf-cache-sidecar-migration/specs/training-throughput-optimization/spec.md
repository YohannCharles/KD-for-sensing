## ADDED Requirements

### Requirement: Multimodal-NF cache migration 诊断
训练吞吐 profile 和并行训练推荐器 MUST 能诊断 Multimodal-NF image/LiDAR 派生 cache 的 sidecar schema 状态，并 MUST 在训练尚未进入 GPU step 前给出可执行的 cache 维护建议。

#### Scenario: profile 输出 sidecar schema 统计
- **WHEN** 用户运行 `conda run -n kd_mm_beam python scripts/profile_training_io.py --config <multimodal-nf-config>`
- **THEN** profile 输出 MUST 包含每个启用 image/LiDAR cache 模态的 sidecar schema version 统计
- **AND** 输出 MUST 包含 valid、migration pending、invalid、missing 和 metadata upgrade supported 数量或等价字段
- **AND** 输出 MUST 标明当前 run 是否可能在进入 GPU step 前执行 cache metadata upgrade、rebuild 或 fallback

#### Scenario: 推荐器发现 migration pending
- **WHEN** 用户对 Multimodal-NF image/LiDAR/fusion 配置运行并行训练推荐器，且 cache `.npy` 存在但 sidecar migration pending 数量大于 0
- **THEN** 推荐器 MUST 输出先运行 derived cache 预处理升级的建议
- **AND** 推荐器 MUST NOT 将 `read_only` 作为唯一推荐策略
- **AND** 推荐器 MUST 说明在 migration pending 清零前，首次训练启动可能主要消耗 CPU/磁盘 IO 而不是 GPU

#### Scenario: 推荐器区分缺失与可迁移 cache
- **WHEN** 推荐器检查 Multimodal-NF image/LiDAR cache 状态
- **THEN** 输出 MUST 区分 cache data missing、sidecar migration pending 和 cache invalid
- **AND** 对 migration pending cache MUST 推荐 metadata-only upgrade 或等价预处理命令
- **AND** 对 missing 或 invalid cache MUST 推荐 rebuild/auto 生成或回退策略

#### Scenario: profile 输出训练阶段状态
- **WHEN** profile 或训练启动诊断发现耗时发生在 dataset/cache 构建阶段
- **THEN** 输出 MUST 以机器可读字段标明尚未进入 GPU step 或 loader iteration
- **AND** 输出 MUST 包含 cache validation/migration 耗时摘要，避免用户只能通过 GPU 利用率猜测问题
