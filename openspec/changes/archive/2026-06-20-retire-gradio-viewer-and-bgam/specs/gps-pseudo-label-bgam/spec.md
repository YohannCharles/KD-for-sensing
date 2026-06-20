## REMOVED Requirements

### Requirement: 历史 GPS pseudo label 生成
**Reason**: GPS pseudo-history label generator 在当前项目中只服务 BGAM reranker。
**Migration**: 无兼容迁移；不得保留 pseudo-history BGAM builder 或配置入口。

### Requirement: mapped label-space pseudo-history 契约
**Reason**: mapped label-space pseudo-history 是 MMW BGAM 主实验的一部分，随 BGAM 退役。
**Migration**: MMW GPS v2 的 label-space/calibration 契约由其自身 specs 继续维护。

### Requirement: 因果时间对齐
**Reason**: 此处定义的是 pseudo-history BGAM 输入时间对齐，不再属于 current workflow。
**Migration**: 保留 workflow 如需因果历史输入，应在对应 capability 中重新定义。

### Requirement: GPS pseudo-history BGAM 输入
**Reason**: BGAM 输入字段、mask source 和 oracle-history 上界均随 BGAM 退役。
**Migration**: 无兼容迁移；保留模型不得依赖 `history_pseudo_*` BGAM 字段。

### Requirement: pseudo-history BGAM 评估产物
**Reason**: pseudo-history diagnostics、mask diagnostics 和 BGAM predictions 随 workflow 退役。
**Migration**: 不再生成或验证 pseudo-history BGAM 产物。
