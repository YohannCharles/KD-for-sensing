## REMOVED Requirements

### Requirement: source-target label histogram 诊断
**Reason**: 该离线诊断产品没有 current CLI、配置或 claim consumer。
**Migration**: 历史 histogram 与结论保留在 archive 和已有 evidence 中。

#### Scenario: 当前工作流不再输出该 histogram
- **WHEN** 用户检查 current 诊断入口
- **THEN** 系统 MUST NOT 暴露 source-target label histogram workflow

### Requirement: 分布距离指标
**Reason**: KL、JS、EMD 和 total-variation 的专用对比仅服务已退役的 distribution-shift workflow。
**Migration**: 如未来出现新的 claim consumer，由独立 change 定义所需的最小指标。

#### Scenario: 专用距离矩阵不再计算
- **WHEN** current evaluation 运行
- **THEN** 它 MUST NOT 要求 distribution-shift 专用距离矩阵

### Requirement: 诊断产物结构
**Reason**: `distribution_shift_metrics.json` 及配套 CSV/图片没有 current consumer。
**Migration**: 历史产物保留在 ignored outputs，不迁移为新源码契约。

#### Scenario: 专用产物不再生成
- **WHEN** current analysis workflow 写出结果
- **THEN** 系统 MUST NOT 要求 distribution-shift 专用 JSON、CSV 或图片

### Requirement: 分布诊断解释字段
**Reason**: absolute/residual shift 解释只属于已退役诊断产品。
**Migration**: 已发布解释保留原有 caveat；current claim 不引用该字段。

#### Scenario: current summary 不消费该字段
- **WHEN** current claim summary 生成
- **THEN** 它 MUST NOT 依赖 absolute/residual distribution-shift 解释字段

### Requirement: Beam distribution diagnostics declare label space
**Reason**: 该 label-space 声明约束随专用 beam-distribution 诊断退役。
**Migration**: 仍在使用的 MMW calibration 和 evaluation 由其 current owner 保持 label-space 契约。

#### Scenario: current owner 独立管理 label space
- **WHEN** current MMW 或 evaluation workflow 使用 calibrated labels
- **THEN** 它 MUST 依据自身 owner spec 记录 label space，而不依赖本 capability
