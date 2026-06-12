## REMOVED Requirements

### Requirement: DeepVerse DT31 generator loads scenario parameters
**Reason**: DeepVerse/DT31 数据生成路线不再属于当前 Image+GPS JEPA query-pool 支持面，本轮删除 generator 和外部 `deepverse` 包入口。
**Migration**: 不迁移到当前 workflow；历史 DeepVerse/DT31 资料只作为归档背景保留，真实数据和本地产物不自动清理。

### Requirement: Phase 1 manifest and labels are generated
**Reason**: DeepVerse/DT31 label builder、manifest、future beam/trajectory/blockage 标签构建退役。
**Migration**: 当前 beam prediction 使用 DeepSense6G/MMW 已准备 split 与 Image+GPS JEPA/query-pool 训练配置。

### Requirement: Radar, weak wireless and noisy position caches are generated
**Reason**: DeepVerse/DT31 专属 cache 生成不再维护，避免继续扩张非主线数据生成面。
**Migration**: 不迁移；如需清理历史 cache，必须使用 runtime cleanup manifest 工作流。

### Requirement: Split and sanity artifacts are generated
**Reason**: DeepVerse/DT31 split 与 sanity workflow 随该数据生成路线整体退役。
**Migration**: 当前 split/sanity 需求由 DeepSense6G/MMW 数据准备和目标场景 split 能力覆盖。
