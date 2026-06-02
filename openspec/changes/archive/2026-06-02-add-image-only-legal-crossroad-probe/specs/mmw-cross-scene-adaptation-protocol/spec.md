## ADDED Requirements

### Requirement: Image-only quick validation eligibility audit
MMW quick validation eligibility audit MUST judge image-only legal probe runs by strict split eligibility and actual target-side oracle consumption. A run MUST NOT be marked ineligible only because raw dataset files or manifest rows contain path、radio、channel、beam_power、GPS、LiDAR or other disabled fields when those fields were not consumed by model input、loss、adaptation、threshold selection、temperature fitting、prototype update、early stopping or summary selection.

#### Scenario: 合法 image-only run 不因原始字段存在被排除
- **WHEN** image-only probe run 使用 strict-validation eligible split
- **AND** consumed fields 只包含 image 输入和允许的 beam labels
- **THEN** eligibility metadata MUST 记录 `target_oracle_fields_used=false`
- **AND** eligibility metadata MUST 记录 `target_radio_label_supervision=false`
- **AND** eligibility metadata MUST 记录 `target_path_label_supervision=false`
- **AND** summary MUST NOT 因 raw dataset 中存在 path、radio、channel 或 beam_power 字段而排除该 run

#### Scenario: 禁用 oracle 实际被消费时排除
- **WHEN** image-only probe run 在 adaptation、threshold selection、temperature fitting、prototype update、early stopping、loss 计算或 summary selection 中消费 target_test label、target_test beam_power、target-side path/radio/channel label 或禁用 oracle 字段
- **THEN** eligibility metadata MUST 将 run 标记为 ineligible
- **AND** `eligibility_reasons` MUST 包含实际字段名、使用阶段和机器可读 reason code
- **AND** summary MUST 将该 run 排除出主结论

### Requirement: Image-only split eligibility 明确化
MMW image-only legal probe MUST 明确记录 split eligibility。系统 MUST NOT 默默输出 `split_eligibility_unknown`；当无法判断 split 是否严格合法时，eligibility metadata MUST 给出缺失 metadata、config path 或 leakage diagnostic path。

#### Scenario: split metadata 完整时可进入主结论
- **WHEN** source、target_support 和 target_test split metadata 能证明 sample id、窗口上下文和 guard-band 约束满足 strict validation
- **THEN** run metadata MUST 记录 `split_eligibility_unknown=false`
- **AND** run metadata MUST 记录 strict split eligibility 的诊断路径或摘要

#### Scenario: split metadata 缺失时给出具体原因
- **WHEN** eligibility checker 无法判断 split eligibility
- **THEN** run metadata MUST 记录 `split_eligibility_unknown=true`
- **AND** `eligibility_reasons` MUST 包含缺失字段、缺失文件或配置路径
- **AND** summary MUST 将该 run 标记为 excluded/debug，而不是把 unknown 当成 eligible

### Requirement: Image-only oracle usage metadata
MMW image-only legal probe MUST 在 run metadata 中记录 enabled modalities、disabled modalities、excluded sensitive fields、consumed fields 和 stage-level oracle usage summary。该 metadata MUST 能支持 downstream report 过滤合法 image-only run。

#### Scenario: metadata 记录模态和禁用字段
- **WHEN** image-only probe run 启动
- **THEN** run metadata MUST 记录 `enabled_modalities=["image"]`
- **AND** run metadata MUST 记录 disabled modalities
- **AND** run metadata MUST 记录 excluded sensitive fields
- **AND** run metadata MUST 记录 `used_target_oracle_fields=[]`，除非实际消费了禁用 target-side oracle 字段

#### Scenario: metadata 记录 stage-level consumed fields
- **WHEN** image-only probe run 完成 source training、target adaptation 或 target_test evaluation stage
- **THEN** run metadata MUST 记录每个 stage 的 consumed input fields 和 consumed label fields
- **AND** target adaptation stage MUST 仅记录 target support image 和 support beam label 作为合法 consumed fields
- **AND** target_test evaluation stage MUST 标记 target_test beam label 仅用于 final metrics

### Requirement: Image-only quick validation conclusion
MMW quick validation summary MUST 为 image-only legal probe 输出机器可读结论。结论 MUST 汇总 eligible run count、ineligible reasons、target oracle flags、split eligibility flags 和各 probe mode 的核心指标。

#### Scenario: eligible run count 大于零
- **WHEN** image-only probe summary 生成
- **THEN** summary MUST 输出 `eligible_run_count`
- **AND** 只有满足 strict split eligibility 且未消费禁用 target oracle 的 run 才能计入 eligible count

#### Scenario: ineligible run 说明可定位
- **WHEN** 任一 image-only probe run 被排除
- **THEN** summary MUST 记录 mode、run directory、eligibility_status、eligibility_reasons、split diagnostics path 和 oracle usage summary
- **AND** reason MUST 足以定位到对应 config path、artifact path 或 stage
