## REMOVED Requirements

### Requirement: JEPA 下游 next-beam query Transformer 主方法
**Reason**: JEPA downstream 主线收敛为 GPS-query attention pooling，不再维护 next-beam query Transformer ablation 作为当前方法。
**Migration**: 使用 `jepa_context_image` + `GPSQueryPool` 的 query-pool 配置，或保留的 GPS-biased/supervised/random-best paired controls。

### Requirement: JEPA downstream ablation 矩阵
**Reason**: `jepa_gru`、`jepa_snapshot`、`jepa_plain_token_transformer` 和 `jepa_next_query_transformer` 配置矩阵退役删除。
**Migration**: 当前 JEPA 配置矩阵只保留 query-pool、GPS-biased baseline、supervised baseline、random-best 控制组和 `beambench_fair` 相关对照配置。

### Requirement: 下游运行 metadata 可追踪
**Reason**: next-query ablation metadata 不再适用于当前支持面。
**Migration**: JEPA downstream metadata 继续由 GPS-query pooling 和 downstream extensibility specs 记录 pooler、checkpoint、adapter 和参数组摘要。

### Requirement: 配置与验证入口
**Reason**: next-beam ablation focused config/test 不再维护。
**Migration**: 使用 query-pool 和 paired control 配置加载/forward 测试。

### Requirement: 不扩大旧研究线
**Reason**: 该约束已由当前 project architecture 和 JEPA query-pool specs 覆盖；next-query 能力本身退役。
**Migration**: 不迁移。
