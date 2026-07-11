## REMOVED Requirements

### Requirement: Scene31 baseline pack run matrix
**Reason**: Scene31 baseline pack 已完成历史对照职责，继续维护独立 local/manual run matrix 会与 final C2 / Scene31-34 当前主线形成重复实验表面。
**Migration**: 当前缺失模态实验使用 final C2、U-Mask eval matrix 和 `scenes31-34-main-missing-modality-workflow`；历史矩阵从 OpenSpec archive 或 git 查询。

#### Scenario: 旧 baseline pack matrix 退出 current surface
- **WHEN** current configs、scripts 和 docs 被枚举
- **THEN** 系统 MUST 不要求 Scene31 baseline pack group 或 run matrix 存在
- **AND** protected final C2、AMR/AMBER supporting contract 和 Scene31-34 main workflow MUST 保持

### Requirement: Random modality dropout training baseline
**Reason**: 该 requirement 只为独立 Scene31 baseline pack 固化一套训练与日志契约，不再是 current 主线的独立能力。
**Migration**: Current missing-modality exposure 由 U-Mask/Scene31-34 current owner 及其配置定义；历史 random-dropout 结果保留在本地产物、archive 和 claim caveat 中。

#### Scenario: baseline-pack random dropout 不再是独立契约
- **WHEN** baseline pack capability 被折叠
- **THEN** current runtime MUST 不为旧 baseline pack 保留专属 dropout group 或 CSV 输出义务
- **AND** 其它 current owner 明确消费的 missing-modality sampling behavior MAY 按其自身 spec 保留

### Requirement: Scene31 baseline pack runner
**Reason**: `scripts/run_scene31_baseline_pack.sh` 是已冻结本地矩阵的专属调度 wrapper，且不属于保留的 package CLI 或 Scene31-34 final owner。
**Migration**: 如需复查历史结果，使用 archive/git 和既有 ignored artifacts；当前训练通过 `kd-sensing-train` 与 protected current configs 执行。

#### Scenario: baseline pack runner 被删除
- **WHEN** 用户查找旧 baseline pack runner
- **THEN** tracked scripts MUST 不再提供该 runner 或同职责 thin wrapper
- **AND** 项目 MUST 不新增 alias、stub 或兼容 launcher

### Requirement: Baseline pack fresh eval 口径
**Reason**: 该 fresh-eval 口径只服务已退役 baseline pack，并依赖本轮同时退出的 apples-to-apples/Scene31 shared summary 表面。
**Migration**: Current missing-pattern evaluation 使用 U-Mask eval matrix 和 Scene31-34 main workflow 的正式评估契约；历史 baseline pack 指标仅作只读证据。

#### Scenario: 旧 baseline pack fresh eval 不再 required
- **WHEN** current evaluation surface 被检查
- **THEN** 系统 MUST 不要求 baseline pack 专属 checkpoint policy、bucket schema 或 sanity check runner
- **AND** current U-Mask/Scene31-34 evaluation schema MUST 不受影响

### Requirement: Baseline pack summary
**Reason**: `scene31_summary --profile baseline-pack` 属于将删除的旧 Scene31 shared summary 产品面。
**Migration**: Current paper-facing summary 使用 protected Scene31-34 final analysis、claim registry 与 paper export；历史 baseline pack 表格从 archive/git 查询。

#### Scenario: baseline-pack summary profile 不再可用
- **WHEN** 用户请求旧 baseline-pack summary profile
- **THEN** current package MUST 不要求该 profile、专属输出文件或 ranking renderer 存在
- **AND** 文档 MUST 不把它列为 current evidence command

### Requirement: Scene31 baseline pack 汇总可接入共享 summary owner
**Reason**: 独立 baseline pack 和旧共享 Scene31 summary owner 同时退役，因此不再需要迁移兼容契约。
**Migration**: 历史汇总实现与 artifact schema 由 git/OpenSpec archive 保存；current Scene31-34 final analysis 保持自己的 owner 和 schema。

#### Scenario: 不建立替代共享 summary facade
- **WHEN** baseline pack summary 被删除
- **THEN** 项目 MUST 不创建新的共享 facade 来维持旧 profile
- **AND** current Scene31-34 final analysis MUST 继续通过其现有 owner 运行
