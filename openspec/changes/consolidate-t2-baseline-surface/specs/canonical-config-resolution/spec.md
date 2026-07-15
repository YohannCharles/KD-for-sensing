## ADDED Requirements

### Requirement: T2/baseline recipes 是唯一 canonical 配置面
配置加载 MUST 将 tracked MMW T2、S1、AMBER-Full 和 RMBP-MM recipe 作为唯一 current canonical experiment inputs。loader MUST 支持这些 recipe 的 shared-base overlay，并 MUST 不从 ignored output、历史 final config 或 virtual broad modality matrix 推导 current recipe。

#### Scenario: T2 recipe 不依赖本地产物
- **WHEN** 用户加载 `configs/mmw/t2.yaml`、`configs/mmw/s1.yaml` 或 `configs/mmw/rmbp_mm.yaml`
- **THEN** 配置 MUST 在无 `outputs/`、checkpoint 和本地数据的环境中完成 parse、normalization 和 validation
- **AND** 最终配置 MUST 声明 MMW 四模态与固定 T2/baseline protocol

## REMOVED Requirements

### Requirement: Config cleanup keeps migration guards
**Reason**: 用户明确要求不保留旧代码兼容；退役路径由删除和历史说明处理。
**Migration**: 使用 tracked T2/baseline recipe；旧路径不提供映射。

### Requirement: Canonical config 解析必须拆分 recipe 与 migration guard
**Reason**: current loader 不再保留 retired-route migration behavior。
**Migration**: 仅加载存在的 tracked recipe 或得到普通缺失配置错误。
