## REMOVED Requirements

### Requirement: AMR-Net_gps_image source audit
**Reason**: AMR-Net_gps_image 不再作为 current source-audit 或 mock/local-substitute workflow 维护；metadata conflict 只保留为 tombstone 背景。
**Migration**: 使用当前 Vision-Position baseline suite、BeamBench/Arnold22 Camera AE+GPS Direct 或其它 current baseline/control。

#### Scenario: source audit runner 退役
- **WHEN** 用户查找 AMR-Net_gps_image source audit runner
- **THEN** 项目 MUST 不提供 current `kd-sensing-run-amr-net-gps-image` 入口

### Requirement: GPS+Image-only modality boundary
**Reason**: 该边界只服务已退役 AMR-Net_gps_image runner。
**Migration**: 当前 GPS+Image 对照使用 Vision-Position baseline suite 或配置化 `modular_sequence`/BeamBench 入口。

#### Scenario: AMR 专属模态边界不再作为 current validator
- **WHEN** 用户加载 AMR-Net_gps_image 专属配置或 override
- **THEN** 系统 MUST 失败或配置不存在

### Requirement: Paper protocol model groups
**Reason**: AMR-Net_gps_image paper groups 不再作为 current workflow 输出。
**Migration**: 使用当前 baseline/control 文档记录的 image-only、GPS-only 或 Image+GPS 对照。

#### Scenario: AMR model groups 不再构建
- **WHEN** 开发者检查 current model registry、preset 或 report runner
- **THEN** 项目 MUST 不要求 AMR-Net_gps_image 专属 model groups 可构建

### Requirement: Paper-aligned metrics and report
**Reason**: 该 report 只服务已退役 AMR-Net_gps_image mock/source-audit runner。
**Migration**: 当前结果记录使用 `docs/result_claims_registry.md` 中仍维护的 claim 行。

#### Scenario: AMR report 不作为 current artifact
- **WHEN** 用户运行当前支持的 baseline 或 diagnostic workflow
- **THEN** 系统 MUST 不要求写出 AMR-Net_gps_image 专属 report

### Requirement: Claim status gating
**Reason**: AMR-Net_gps_image 不再保留 current claim 行；official reproduction block 只作为历史/tombstone 说明。
**Migration**: 若未来重新开展 AMR-Net 复现，必须另起 OpenSpec change 并重新声明 claim gating。

#### Scenario: AMR claim 行移出 current registry
- **WHEN** 开发者阅读 result claim registry
- **THEN** AMR-Net_gps_image MUST 不作为 current pending、mock-smoke 或 blocked official claim 行出现

### Requirement: Runtime artifact boundary
**Reason**: AMR-Net_gps_image runner 退役后不再生成 current runtime artifacts。
**Migration**: 历史本地产物只能作为 archive 背景或 cleanup manifest 候选。

#### Scenario: AMR runtime root 不作为 current output
- **WHEN** 开发者阅读 README 或 project inventory
- **THEN** `outputs/analysis/amr_net_gps_image/` MUST 不作为 current workflow 默认输出根推荐

## ADDED Requirements

### Requirement: AMR-Net_gps_image retired tombstone
项目 MUST 将 AMR-Net_gps_image / IEEE `11282996` 视为已退役 source-audit/local-substitute workflow。源码、配置、CLI、测试和结果账本 MUST 不再把该 workflow 暴露为 current entry point；历史 metadata conflict MAY 保留为 tombstone 说明。

#### Scenario: AMR 入口和配置不存在
- **WHEN** 开发者检查 pyproject、包内 CLI、configs 和 baselines package
- **THEN** 项目 MUST 不声明 `kd-sensing-run-amr-net-gps-image`
- **AND** 项目 MUST 不保留 `configs/baselines/amr_net_gps_image.yaml` 作为 current config
- **AND** 项目 MUST 不保留 `kd_sensing.baselines.amr_net_gps_image` 作为 current workflow package

