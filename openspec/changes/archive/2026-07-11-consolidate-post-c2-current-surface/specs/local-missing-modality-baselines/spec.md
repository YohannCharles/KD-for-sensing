## ADDED Requirements

### Requirement: Local missing-modality capability 只保留 AMR-lite supporting contract
该 capability MUST 分类为 supporting，并只保护 Scene31-34 runner 与 generated configs 仍消费的 AMR-lite component、missing-mask/gate 和 local-baseline claim boundary。AMBER-lite、AMBER full 与 AMR-Net MUST 分别由其独立 current specs 管理；legacy baseline-pack、FeatureMod 和旧 maskfix wrappers MUST 不借此恢复。

#### Scenario: Scene31-34 构建 AMR-lite
- **WHEN** protected Scene31-34 generator 生成 `amr_lite` natural/uniform config
- **THEN** current component registry MUST 能构建该 local AMR-lite baseline
- **AND** missing modality MUST 不泄漏真实 feature，availability mask MUST 参与 gate/attention

#### Scenario: Supporting lifecycle 不恢复 baseline pack
- **WHEN** lifecycle、scripts 和 configs 被枚举
- **THEN** capability MUST 标记为 `supporting`
- **AND** FeatureMod、旧 Scene31 baseline pack、RMBP/TII 聚合入口和 retired maskfix wrappers MUST 不存在

## MODIFIED Requirements

### Requirement: AMR-lite local baseline
系统 MUST 为 protected Scene31-34 external-lite workflow 保留 AMR-lite local missing-modality component，用于表达 available modality masking、missing modality imputation 和轻量 channel/modality attention。该 baseline MUST 标记为 local experimental baseline，不得声明为完整外部论文复现，也不得扩大为 standalone baseline-pack product。

#### Scenario: AMR-lite 构建
- **WHEN** Scene31-34 generated config 声明 `amr_lite`
- **THEN** 系统 MUST 能构建支持 image、LiDAR、radar 和 GPS modality feature 的轻量 fusion component
- **AND** 缺失 modality MUST 使用 current config 声明的 imputation，availability/missing mask MUST 参与 gate 或 attention

#### Scenario: AMR-lite 防泄漏
- **WHEN** 某 modality 在训练或评估 batch 中被标记为缺失
- **THEN** AMR-lite MUST NOT 使用该 modality 的真实 feature 作为可见输入
- **AND** imputation 和 gate 计算 MUST 只依赖缺失 token、可见 feature 和 mask metadata

#### Scenario: AMR-lite diagnostics 服从 Scene31-34 owner
- **WHEN** AMR-lite 进入 Scene31-34 fresh evaluation
- **THEN** output MUST 使用 Scene31-34 current maskfix/status contract
- **AND** capability MUST 不要求独立 epoch gate CSV 或 standalone diagnostics runner

## REMOVED Requirements

### Requirement: 本地可训练 baseline 入口
**Reason**: 该聚合要求把 RMBP-MM、TII-style 与 AMBER 混成一个长期 product；current configs 已由各自 owner或 H5/P1/Scene31-34 消费。
**Migration**: 使用各 current model spec 与 canonical `kd-sensing-train` 入口。
#### Scenario: 聚合入口退出
- **WHEN** local baseline surface 被枚举
- **THEN** 项目 MUST 不维护跨家族聚合入口或配置包

### Requirement: 缺失模态训练扰动接入
**Reason**: 与 modality difficulty、U-Mask 和 Scene31-34 current contracts 重复。
**Migration**: 使用统一 difficulty pipeline owner。
#### Scenario: 扰动契约不重复
- **WHEN** AMR-lite/AMBER-lite 训练应用 missing profile
- **THEN** behavior MUST 由 current difficulty owner验证

### Requirement: baseline claim 边界
**Reason**: AMBER-lite、AMBER full、AMR-Net 和 Scene31-34 specs 已分别记录 claim caveat。
**Migration**: 使用对应独立 owner。
#### Scenario: Claim owner 唯一
- **WHEN** current summary 描述 external-lite baseline
- **THEN** claim caveat MUST 来自其独立 spec 或 Scene31-34 owner

### Requirement: AMBER full 纳入本地缺失模态 baseline 家族
**Reason**: `amber-full-architecture-reproduction` 已是独立 current owner。
**Migration**: 使用该 capability。
#### Scenario: AMBER full 不重复归属
- **WHEN** AMBER full 被构建
- **THEN** 本 capability MUST 不维护其第二份要求

### Requirement: Local baseline stress comparability
**Reason**: Missing-modality stress suite 与各 model owner 已维护 strict comparability。
**Migration**: 使用 `missing-modality-stress-suite`。
#### Scenario: Stress metadata owner 唯一
- **WHEN** local baseline 进入 stress suite
- **THEN** comparability MUST 由 stress owner验证

### Requirement: AMBER-lite baseline-pack training comparison
**Reason**: Scene31 legacy baseline pack 已退役；protected Scene31-34 runner 已明确当前 AMBER-lite groups。
**Migration**: 使用 Scene31-34 external-lite groups。
#### Scenario: Baseline pack matrix 退出
- **WHEN** current AMBER-lite configs 被生成
- **THEN** 项目 MUST 不要求旧 baseline-pack seed matrix

### Requirement: FeatureMod-lite local baseline
**Reason**: FeatureMod-lite 没有 current config/source consumer，且旧 baseline pack 已退出。
**Migration**: 无；未来重启必须新 change。
#### Scenario: FeatureMod 不构建
- **WHEN** config 请求 `featuremod_lite`
- **THEN** registry MUST 返回 unknown/removed failure

### Requirement: Baseline pack 参数量比较
**Reason**: 旧 baseline-pack summary 退役；current Scene31-34 profile 已有 compute/parameter owner。
**Migration**: 使用 protected Scene31-34 profile。
#### Scenario: 参数表不重复
- **WHEN** Scene31-34 导出 compute profile
- **THEN** 项目 MUST 不要求旧 `params_comparison.csv`

### Requirement: Modular-lite missing-mask fresh eval diagnostics
**Reason**: Current Scene31-34 spec 已保护 maskfix fresh eval、suspect exclusion 与 official ranking。
**Migration**: 使用 consolidated Scene31-34 owner。
#### Scenario: 旧 diagnostics wrapper 退出
- **WHEN** current scripts 被枚举
- **THEN** standalone modular missing-mask diagnostic MUST 不存在

### Requirement: Modular-lite formal maskfix fresh eval artifacts
**Reason**: Artifact schema 与 protected Scene31-34 fresh eval/summary 要求重复。
**Migration**: 使用 Scene31-34 maskfix-marked root 与 current summary fields。
#### Scenario: Maskfix artifact owner 唯一
- **WHEN** AMR/AMBER-lite 被 fresh-eval
- **THEN** artifact MUST 由 Scene31-34 owner写出

### Requirement: Modular-lite mask suspect artifact
**Reason**: Scene31-34 current spec 已要求 `mask_suspect` 控制 official ranking/checklist。
**Migration**: 使用 Scene31-34 current summary/evidence gate。
#### Scenario: Suspect 判定不重复
- **WHEN** external-lite result 被汇总
- **THEN** suspect/exclusion MUST 由 Scene31-34 owner解释
