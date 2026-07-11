## REMOVED Requirements

### Requirement: Scene31 next-round 配置矩阵
**Reason**: Scene31 next-round P0/condBTAPA 矩阵是已冻结的 local/manual 搜索面，不属于 final C2 / U-MaskBeamJEPA 当前主线。
**Migration**: Current temporal 与 missing-modality 实验使用 final C2、H5/P1 和 Scene31-34 main owners；旧矩阵从 OpenSpec archive 或 git 查询。

#### Scenario: next-round matrix 退出
- **WHEN** current configs、manifest 和 inventory 被枚举
- **THEN** 项目 MUST 不要求旧 next-round run names、generator 或 generated YAML 存在
- **AND** protected current configs 和用户 H5/P1 launcher 改动 MUST 保持

### Requirement: Scene31 P0 fresh eval runner
**Reason**: P0 runner 只调度已退役 next-round 矩阵，并依赖本轮同时退出的 apples-to-apples helper。
**Migration**: Current missing-pattern evaluation 使用 U-Mask eval matrix 和 Scene31-34 main fresh-eval owner；历史 P0 结果只读保留。

#### Scenario: P0 runner 不再 required
- **WHEN** current scripts 被检查
- **THEN** 项目 MUST 不提供旧 P0 fresh-eval runner 或同职责 wrapper
- **AND** current evaluation owners MUST 不依赖该脚本

### Requirement: Scene31 next-round 汇总
**Reason**: 该汇总 schema 只服务已退役 next-round run family，与 protected final analysis 重复。
**Migration**: Current paper-facing summary 使用 Scene31-34 final analysis、claim registry 和 paper export；历史表格从 archive/git 查询。

#### Scenario: next-round summary 退出
- **WHEN** current reporting surface 被枚举
- **THEN** 系统 MUST 不要求旧 next-round per-run、mean/std、delta 或 top10 artifacts
- **AND** protected final analysis schemas MUST 保持

### Requirement: Scene31 BC next-round matrix
**Reason**: adaptive sampler、beamsoft 和 combined BC 搜索矩阵是未晋级的历史候选面。
**Migration**: Current baseline/loss behavior 仅由其独立 current specs 与 final C2 configs 定义；旧 BC matrix 留在 archive/git。

#### Scenario: BC matrix 不再生成
- **WHEN** config generator 被运行或审计
- **THEN** generator MUST 不要求生成旧 BC P0/P1 run family
- **AND** 系统 MUST 不恢复 condBTAPA/weakKD 历史组合

### Requirement: Scene31 BC launcher
**Reason**: BC launcher 是只服务退役矩阵的 local/manual orchestration wrapper。
**Migration**: Current training 使用 `kd-sensing-train` 和明确保留的 scripts；历史 BC 调度方式从 git 查询。

#### Scenario: BC launcher 被删除
- **WHEN** current scripts inventory 被检查
- **THEN** `run_scene31_bc_next` 或等价 thin wrapper MUST 不存在
- **AND** 项目 MUST 不新增兼容 alias 或 stub

### Requirement: Scene31 BC summary
**Reason**: BC summary 的固定 reference 数值和排序只服务已冻结候选搜索，不属于 current claim contract。
**Migration**: Current result comparison 使用 formal claim/protocol、Scene31-34 final evidence 和 paper export gate。

#### Scenario: BC summary schema 退出
- **WHEN** current summary owners 被枚举
- **THEN** 系统 MUST 不要求旧 BC summary、fixed uniform winner 或专属 delta 字段
- **AND** current claim comparability MUST 继续由正式 owner 校验

### Requirement: Scene31 magic overnight matrix
**Reason**: magic overnight 矩阵是历史候选搜索 batch，不属于 current config surface。
**Migration**: 旧 run names、method tags 和结果通过 archive/git/ignored artifacts 查询；不迁移到新 launcher framework。

#### Scenario: magic overnight matrix 退出
- **WHEN** tracked Scene31 configs 与 manifests 被检查
- **THEN** 系统 MUST 不要求 magic overnight generator、manifest 或 configs
- **AND** final C2/H5-P1 protected matrix MUST 不受影响

### Requirement: Scene31 magic overnight 4 GPU runner
**Reason**: 固定四卡 local runner 只服务已退役 magic overnight matrix，并复制用户本地调度职责。
**Migration**: Current long-running work 使用用户 shell、tmux、任务系统或保留的 parameterized scripts，不建立替代 queue wrapper。

#### Scenario: magic runner 不再存在
- **WHEN** scripts surface 被枚举
- **THEN** `run_scene31_magic_overnight` runner 和专属 failed-list contract MUST 不再是 current surface
- **AND** current package CLI MUST 不依赖该 runner

### Requirement: Missing-pattern DRO training
**Reason**: 该 MP-DRO requirement 作为 Scene31 next-round 候选训练分支提出，未晋级为 final C2 current contract。
**Migration**: U-MaskBeamJEPA 保留分支以其 current spec 为准；未来若需要 MP-DRO，必须由新的 OpenSpec change 重新定义，而不是保留旧 workflow 契约。

#### Scenario: MP-DRO 不再是 current obligation
- **WHEN** current model、loss 和 training config surface 被审计
- **THEN** 系统 MUST 不要求 `training.mpdro`、EMA group weights 或专属 CSV 日志存在
- **AND** protected U-Mask fusion/loss branches MUST 不因该 requirement 删除而受损

### Requirement: Scene31 funnel missing bucket summary
**Reason**: funnel missing-bucket summary 只为已退役 funnel candidate selection 提供中间统计。
**Migration**: Current missing-pattern grouping 与 evidence 使用 U-Mask eval matrix、missing-modality statistics/stress 和 Scene31-34 final analysis。

#### Scenario: funnel bucket summary 退出
- **WHEN** current evaluation/reporting surface 被检查
- **THEN** 系统 MUST 不要求旧 funnel bucket mapping 或专属 summary artifact
- **AND** current missing-pattern statistics MUST 保持其自身 schema

### Requirement: Scene31 missing-aware checkpoint selection
**Reason**: 该 checkpoint promotion rule 绑定已退役 funnel 搜索，不能继续覆盖 current training runtime 的通用 checkpoint 契约。
**Migration**: Current checkpoint selection 由 training/evaluation runtime 和具体 final C2 protocol 定义；历史 selection 结果仅作只读记录。

#### Scenario: funnel checkpoint rule 退出
- **WHEN** current checkpoint policies 被解析
- **THEN** 系统 MUST 不要求旧 Scene31 missing-aware composite score 或 promotion artifact
- **AND** current best-checkpoint/resume semantics MUST 保持

### Requirement: Scene31 funnel local/manual matrix and runner
**Reason**: funnel matrix/runner 是未晋级候选的本地编排面，已完成筛选职责。
**Migration**: Current experiments 使用 protected final C2、H5/P1 和 Scene31-34 main workflows；历史 funnel runs 从 ignored artifacts/archive 查询。

#### Scenario: funnel runner 退出
- **WHEN** scripts、configs 和 inventory 被枚举
- **THEN** 项目 MUST 不要求 funnel matrix、runner groups 或 generated configs
- **AND** 不得创建同职责兼容 wrapper

### Requirement: Scene31 funnel summary and conservative conclusion
**Reason**: funnel summary/ranking 只服务已退役 candidate promotion loop。
**Migration**: Current conclusions 由 formal claim registry、Scene31-34 final analysis 与 paper export gate 维护。

#### Scenario: funnel conclusion 不再生成
- **WHEN** current report commands 被检查
- **THEN** 系统 MUST 不要求 funnel profile、promotion labels 或专属 conclusion artifact
- **AND** current claim caveat 与 provenance MUST 保持

### Requirement: Scene31 mild MP-DRO training logs
**Reason**: mild MP-DRO 日志 schema 绑定未晋级 funnel ablation，不是 current runtime schema。
**Migration**: 若未来 current method 需要 group-DRO 日志，必须由其 owner spec 重新定义；历史 CSV schema 留在 archive/git。

#### Scenario: mild MP-DRO 日志义务退出
- **WHEN** current training logs 被校验
- **THEN** 系统 MUST 不要求 `raw_weight`、`protected_weight` 或旧 MP-DRO CSV 字段
- **AND** current training log 与 TensorBoard schema MUST 保持

### Requirement: Scene31 manifest-backed workflow 必须保持生成与运行分离
**Reason**: 该 requirement 只治理 next-round/BC/funnel/magic 等整体退役的 manifest family。
**Migration**: Current config/manifest generation 继续按各自保留 owner 的 specs 管理；本 change 不建立通用 Scene31 launcher framework。

#### Scenario: 旧 manifest family 不再受 current contract 保护
- **WHEN** retired Scene31 manifests 和 generators 被删除
- **THEN** current specs MUST 不要求其生成字段或 runner integration 保持兼容
- **AND** protected Scene31-34/final C2 manifests MUST 不被删除

### Requirement: Scene31 PatternFiLM d8 follow-up workflow
**Reason**: PatternFiLM d8 是未晋级的 focused follow-up，继续保留会维持一整套专属 config/eval/summary surface。
**Migration**: 历史 PatternFiLM 结果和 caveat 从 archive/git 查询；future revival 需要新的 OpenSpec change。

#### Scenario: PatternFiLM follow-up 退出
- **WHEN** current configs、registry 和 scripts 被检查
- **THEN** 系统 MUST 不要求旧 PatternFiLM d8 run family、fresh eval 或 promotion rules
- **AND** final C2 current branches MUST 保持

### Requirement: Scene31 subset reference summary
**Reason**: `proto_randomdrop_subset_es40` reference 选择绑定已退役 subset comparison loop，不应继续作为项目级 current winner reference。
**Migration**: Current reference 与 claim comparability 由 final C2/Scene31-34 formal protocol 和 claim registry 定义。

#### Scenario: subset reference 不再是 current default
- **WHEN** current ranking 或 claim tables 被生成
- **THEN** 系统 MUST 不要求使用旧 subset run 或 fixed fallback values 作为默认 reference
- **AND** historical comparison MAY 在明确 historical 语境中保留

### Requirement: Scene31 subset reliability and PatternFiLM workflow
**Reason**: subset reliability/PatternFiLM runner 是未晋级支线，且与保留的 Scene31-34 main workflow 重复。
**Migration**: Current multi-scene/final validation 使用 Scene31-34 main owner 和 U-Mask eval matrix；历史支线从 archive/git 查询。

#### Scenario: subset reliability workflow 退出
- **WHEN** scripts 和 generated config groups 被枚举
- **THEN** 项目 MUST 不要求 maskfix、reliability 或 subset_film 专属 runner group
- **AND** current AMR/AMBER supporting models MAY 继续由其 own specs 维护

### Requirement: Scene31 subset combined summary
**Reason**: combined summary 只聚合已退役 subset candidate families，并维护一套重复 ranking/promotion schema。
**Migration**: Current final evidence aggregation 使用 Scene31-34 final analysis 和 paper export；历史 combined summary 留在 archive/git。

#### Scenario: combined subset summary 退出
- **WHEN** current summary profiles 被检查
- **THEN** 系统 MUST 不要求 subset-reliability profile、delta table 或 promotion labels
- **AND** formal current evidence outputs MUST 保持

### Requirement: Scene31 subset summary prefers modular maskfix results
**Reason**: maskfix precedence 与 suspect-row ranking 只服务退役 subset summary；AMR/AMBER current behavior 已由各自 owner 和 focused tests 覆盖。
**Migration**: AMR-lite/AMBER-lite 的 current evaluation contract 以独立 capability specs 和 Scene31-34 protected consumers 为准。

#### Scenario: maskfix ranking policy 退出
- **WHEN** old subset summary 被删除
- **THEN** 系统 MUST 不为其保留 `fresh_eval_maskfix` precedence 或 suspect ranking schema
- **AND** protected AMR/AMBER model behavior MUST 不被删除

### Requirement: Scene31 reliability seed continuation runner
**Reason**: reliability seed3/4/5 continuation 是已冻结的 local candidate expansion loop。
**Migration**: 历史 seed evidence 保留在 ignored artifacts/claim notes；future continuation 需要新的 scoped change。

#### Scenario: seed continuation groups 退出
- **WHEN** current runner groups 被枚举
- **THEN** 系统 MUST 不要求 `reliability_seed3`、`reliability_seed45` 或 overwrite-failed orchestration
- **AND** 不得把历史 failed run 转换为 current workflow 义务

### Requirement: Scene31 reliability promotion status
**Reason**: promotion gate 绑定退役 subset reference 和 reliability candidate，不再代表 current claim gate。
**Migration**: Current promotion/readiness 由 formal claim registry、protocol 和 paper export gate 决定。

#### Scenario: reliability promotion label 退出
- **WHEN** current claim status 被计算或更新
- **THEN** 系统 MUST 不要求 `candidate_continue_to_seed5` 或 `do_not_expand_now` 旧标签
- **AND** formal claim status MUST 保留人工审阅 provenance

### Requirement: Scene31 next-round summary 可由共享 owner 产出
**Reason**: next-round profiles 和旧共享 Scene31 summary owner 同时退出，不需要保持 profile 兼容层。
**Migration**: Current Scene31-34 final analysis 使用其现有 owner；历史 summary profiles 通过 archive/git 查询。

#### Scenario: 不建立 replacement summary facade
- **WHEN** next-round summary 实现被删除
- **THEN** 项目 MUST 不创建只设置固定 profile 的新 wrapper 或 facade
- **AND** current final analysis owner MUST 继续独立运行

### Requirement: Scene31 summary 删除必须防止脚本回流
**Reason**: 专用 tombstone capability 被折叠后，不再为旧 Scene31 summary 维护独立 guard requirement。
**Migration**: 同名/同职责 wrapper 的防回流语义迁入 `retired-route-summary` 与轻量 architecture guard。

#### Scenario: 集中 guard 拒绝 wrapper 回流
- **WHEN** tracked scripts 新增只转发旧 Scene31 summary profile 的 wrapper
- **THEN** centralized retired-route/architecture guard MUST 报告该回流
- **AND** 项目 MUST 不恢复本 capability 作为专用 tombstone
