## MODIFIED Requirements

### Requirement: Image observability transform
系统 MUST 提供通用 image observability transform，用于在统一 difficulty pipeline 中确定性调制 image modality。Transform MUST 支持 frame dropout、burst dropout、weather、blur、occlusion、low-light、seed、valid mask、observability score 和 replay metadata；它 MUST 通过普通 profile/operator parameters 配置，不要求已退役 Scenario-D preset。

#### Scenario: ImageObservabilityTransform 参数
- **WHEN** 开发者通过 current difficulty profile 构建 image observability transform
- **THEN** transform MUST 支持 `image_dropout_prob`、`image_burst_dropout_prob`、`max_burst_len`、`image_weather_severity`、`image_blur_prob`、`image_occlusion_prob`、`image_occlusion_ratio`、`image_lowlight_prob` 和 `seed`
- **AND** 参数 MUST 由普通 difficulty profile/operator config 标准化生成

#### Scenario: corruption 与 missing 区分
- **WHEN** transform 只应用 weather、low-light、blur 或 occlusion
- **THEN** `image_valid_mask` MUST 保持有效，除非配置另行声明整帧不可用
- **AND** transform MUST 写出 corruption type、severity、frame range 和 operator parameters
- **AND** GPS、target label、sample id 和 split metadata MUST 保持不变

#### Scenario: dropout 生成 invalid mask
- **WHEN** transform 应用 frame dropout 或 burst missing
- **THEN** 系统 MUST zero-fill、mask-fill 或使用配置声明的 missing token 表达缺失 image
- **AND** 系统 MUST 写出 `image_valid_mask`、`image_dropout_mask` 或 `image_burst_dropout_mask`

## REMOVED Requirements

### Requirement: Scenario D preset 复用 difficulty pipeline
**Reason**: Scenario-D image observability benchmark 已退役，没有 current config/runner consumer。
**Migration**: 普通 image degradation 与 missing-stress 继续使用通用 operator parameters。
#### Scenario: D-level preset 退出
- **WHEN** config 请求旧 D0-D7 preset
- **THEN** current parser MUST 不恢复该命名 suite

### Requirement: Predictive robustness difficulty preset
**Reason**: P0-P5 只服务已退役 Predictive GPS-query++ benchmark。
**Migration**: Current temporal missing 和 JEPA pretraining 使用各自 owner。
#### Scenario: P-level preset 退出
- **WHEN** config 请求 `predictive_jepa_robustness`
- **THEN** parser MUST 返回 unknown/removed failure

### Requirement: History-aware image missing 和 semantic occlusion
**Reason**: 该专属 operator contract 只服务 P-level predictive benchmark。
**Migration**: 通用 frame/burst missing 与 image occlusion operators 保留。
#### Scenario: Predictive semantic branch 不存在
- **WHEN** current image difficulty operator 运行
- **THEN** 它 MUST 不要求 predictive condition 或 history-availability schema

### Requirement: Plausible wrong GPS 扰动
**Reason**: Peer-sample counterfactual GPS replacement 只服务 retired predictive/GPS-query evidence。
**Migration**: 通用 jitter、drift、delay、dropout 和 distractor operators 保留。
#### Scenario: Peer GPS replacement 退出
- **WHEN** current GPS profile 被解析
- **THEN** parser MUST 不要求 predictive peer pool 或 beam-offset selection

### Requirement: Predictive robustness determinism 和 no-future-leak
**Reason**: Predictive suite 整体删除；专属 replay schema 没有 consumer。
**Migration**: 通用 GPS temporal operators 继续遵守 no-future-leak 与 determinism requirements。
#### Scenario: 通用 no-future-leak 保持
- **WHEN** current GPS delay/low-rate operator 运行
- **THEN** source index MUST 继续不晚于当前时间步
- **AND** 不要求 predictive replay metadata

### Requirement: Visual-ambiguous hard negative condition
**Reason**: Visual peer selection 仅服务 GPS-query advantage evaluation。
**Migration**: 无；未来 hard-negative work 需新 change。
#### Scenario: Visual hard negative 退出
- **WHEN** profile 请求 visual-ambiguous peer selection
- **THEN** current parser MUST 拒绝该 retired condition

### Requirement: Beam-offset-constrained wrong GPS
**Reason**: Beam-aware peer GPS selection 仅服务 retired GPS-query reliance benchmark。
**Migration**: 普通 GPS difficulty operators 不读取 target beam 选择 peer。
#### Scenario: Beam-offset peer selection 退出
- **WHEN** current GPS operator 构建
- **THEN** 它 MUST 不要求 target-beam peer pool

### Requirement: Combined GPS-query advantage perturbations
**Reason**: CxD advantage matrix 与 GPS-query benchmark 已退役。
**Migration**: Current profiles仍可显式组合普通 GPS/image operators，不保留 CxD suite naming。
#### Scenario: CxD preset 退出
- **WHEN** config 请求旧 GPS-query CxD advantage condition
- **THEN** parser MUST 拒绝该 preset

### Requirement: Advantage difficulty determinism
**Reason**: 对应 advantage conditions 被删除，专属 peer-id determinism 不再有对象。
**Migration**: 通用 difficulty determinism requirement 保持。
#### Scenario: 通用 determinism 承接
- **WHEN** current profile 以同 seed/sample 重放
- **THEN** ordinary operator output MUST 继续确定

### Requirement: Difficulty perturbation cache provenance
**Reason**: 该 cache 只服务已退役 benchmark reuse，当前没有 source consumer。
**Migration**: Runtime cache 边界仍由一般 artifact policy 管理。
#### Scenario: 旧 perturbation cache 退出
- **WHEN** manifest 请求旧 benchmark cache payload
- **THEN** current difficulty pipeline MUST 不提供该 cache schema

### Requirement: Benchmark 复用统一 difficulty pipeline
**Reason**: JEPA GPS shortcut benchmark 与其 legacy suite adapter 已退役。
**Migration**: Current train/evaluation/missing-stress 直接使用统一 difficulty pipeline。
#### Scenario: Legacy suite adapter 退出
- **WHEN** manifest 请求旧 shortcut suite type 或 Scenario-C preset
- **THEN** current pipeline MUST 不保证旧 metrics/report compatibility

### Requirement: Benchmark 输出 difficulty provenance
**Reason**: 专属 shortcut benchmark output schema 随 runner 退出。
**Migration**: Current run/evaluation 继续记录 profile digest、operator、seed 和 warnings。
#### Scenario: Current provenance 不依赖 benchmark manifest
- **WHEN** current train/evaluation 使用 difficulty profile
- **THEN** run metadata MUST 继续可审计
- **AND** 不要求旧 `benchmark_manifest.json`
