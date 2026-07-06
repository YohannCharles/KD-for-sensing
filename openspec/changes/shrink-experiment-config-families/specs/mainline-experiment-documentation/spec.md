## ADDED Requirements

### Requirement: 实验文档必须跟随配置生成化
当 experiment config family 的实体 YAML 被删除、生成化或降级为 local/manual 后，主线模型目录、实验矩阵、协议表和 result claim registry MUST 指向真实存在的 current config、generator/manifest 输入、base config 或明确 historical/local/manual 说明。

#### Scenario: experiment matrix 不指向删除配置
- **WHEN** 开发者阅读 `docs/experiment_matrix.md` 或 `docs/mainline_model_catalog.md`
- **THEN** current workflow 行 MUST 不引用已删除实体 YAML 作为可运行入口
- **AND** 若复跑需要生成配置，文档 MUST 指向 generator、manifest、base config 和输出边界

#### Scenario: claim provenance 保持可审计
- **WHEN** claim 或 pending claim 依赖的 YAML 被生成化或删除
- **THEN** `docs/result_claims_registry.md` 或等价 provenance MUST 更新为保留 YAML、generator/manifest 输入或 historical caveat
- **AND** pending/mock/local/manual 状态 MUST 不因删除实体 YAML 而被提升为 promoted claim

### Requirement: 配置族状态必须出现在主线文档
主线文档 MUST 区分 current recommended config、paper/workflow reproduction config、secondary/supporting config、local/manual overlay、generated config 和 historical config。读者 MUST 能从文档判断一个配置是否适合直接运行、是否需要本地 checkpoint/outputs、以及是否支撑当前 claim。

#### Scenario: local/manual 配置不冒充 current
- **WHEN** 文档列出 Scene31、RBMA、strong encoder 或 JEPA image+GPS local/manual 配置
- **THEN** 文档 MUST 标记其 local/manual 或 pending 状态、输出边界和主要 caveat
- **AND** 文档 MUST 不把缺少 evidence 或依赖本地 checkpoint 的配置描述为 promoted mainline result

#### Scenario: generated 配置有复跑路径
- **WHEN** 文档描述一个不再跟踪实体 YAML 的 generated config family
- **THEN** 文档 MUST 给出 generator/manifest/base config 或 package CLI 复跑路径
- **AND** 文档 MUST 说明生成产物默认进入 ignored local output/config directory
