## ADDED Requirements

### Requirement: CSI hardening matrix 可由 base config 和 overlay recipe 表达
CSI hardening matrix MUST 保持 A/B/C/D/E 组逻辑配置 ID 可加载和可审计，但系统 MAY 使用 base config、overlay YAML、recipe table 或现有配置解析机制生成 resolved config。项目 MUST 不要求每个矩阵 ID 长期维护一份重复完整 YAML 文件。

#### Scenario: CSI-only 配置 ID 仍可加载
- **WHEN** 用户或测试加载 A0、A1、A2、B3、B4、B5、B6、C1、C2、D1、D2、D3 或 D4 hardening matrix 配置 ID
- **THEN** 系统 MUST 解析出等价的 `modular_sequence`、`pilot_dual_view_csi` 和 beam prediction 输出契约
- **AND** resolved config MUST 记录足以追踪 base config、overlay/recipe ID 和关键控制变量的 metadata

#### Scenario: destructive negative control 语义保持
- **WHEN** 用户加载 A2 destructive degradation negative control
- **THEN** resolved config MUST 显式启用 `data.dataset.csi_degradation`
- **AND** resolved config MUST 不启用 information-preserving `csi_hardening`

#### Scenario: D 组不启用 destructive degradation
- **WHEN** 用户加载 D1、D2、D3 或 D4 组合配置
- **THEN** resolved config MUST 表达对应 hardening 和架构组合
- **AND** resolved config MUST 不启用 destructive `csi_degradation`

#### Scenario: 多模态验证配置语义保持
- **WHEN** 用户加载 E0、E1、E2 或 E3 easy modality + CSI 验证配置
- **THEN** resolved config MUST 保持 GPS-only、GPS+clean CSI、GPS+slow CSI 和 GPS+slow CSI warmup 的逻辑差异
- **AND** E1 到 E3 MUST 使用 `modalities: [gps, csi]` 或等价归一化后的模态集合

#### Scenario: 测试不逐行冻结重复 YAML
- **WHEN** 架构边界测试或配置加载测试验证 CSI hardening matrix
- **THEN** 测试 MUST 验证配置 ID、关键 resolved 字段、控制变量和 destructive/hardening 边界
- **AND** 测试 MUST 不要求每个配置 ID 都对应一份完整实体 YAML
