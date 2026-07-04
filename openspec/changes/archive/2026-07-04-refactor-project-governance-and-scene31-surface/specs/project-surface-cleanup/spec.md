## ADDED Requirements

### Requirement: Generated Scene31 YAML 必须可清理或有理由保留
Scene31 generated YAML MUST 不静默扩大源码表面。需要长期跟踪的实体 YAML MUST 有 current/local/manual 保留理由；可由 generator 和 manifest 无损重建的 YAML MUST 改为本地生成产物或登记删除计划。

#### Scenario: 可再生成 YAML 不长期堆积
- **WHEN** generator 能从 template 和 manifest 重建 Scene31 YAML
- **THEN** 源码表面积治理 MUST 优先保留 generator、manifest 和 template
- **AND** 实体 YAML 若继续跟踪，MUST 说明不可由 generator 无损重建的字段或人工样例价值

#### Scenario: 清理不触碰运行产物
- **WHEN** 清理 Scene31 源码配置表面
- **THEN** 实现 MUST 不删除、移动或重写 `outputs/scene31*`、`logs/`、checkpoint、fresh eval 结果或本地 cache
