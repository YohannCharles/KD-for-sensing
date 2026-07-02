# csi-hardening-experiment-matrix Specification

## Purpose
定义 CSI hardening 实验矩阵的控制变量、配置命名和解释边界，保证对照实验、输出目录和结论归因遵循同一契约。
## Requirements
### Requirement: CSI hardening 控制变量配置矩阵
系统 MUST 提供可加载的 CSI hard-to-learn 控制变量配置矩阵，用于比较 clean CSI、destructive degradation、information-preserving hardening、CSI encoder 架构消融以及 hardening+架构组合。配置 MUST 固定 beam prediction 任务、MMW Town10 skybridge 场景、100 epoch 默认训练长度和可覆盖的 seed。

#### Scenario: CSI-only 第一批配置可加载
- **WHEN** 用户加载 CSI-only hardening matrix 的第一批配置
- **THEN** 系统 MUST 至少提供 A0 clean full teacher、A1 mild pilot estimation、A2 destructive degradation negative control、B3 antenna calibration、B4 fixed antenna permutation、B5 mild hardening combo、B6 medium hardening combo、C1 view gate warmup 和 C2 no internal GRU 配置
- **AND** 每个配置 MUST 使用 `modular_sequence`、`pilot_dual_view_csi` 和 beam prediction 输出契约

#### Scenario: CSI-only 第二批组合配置可加载
- **WHEN** 用户加载 CSI-only hardening matrix 的组合配置
- **THEN** 系统 MUST 至少提供 D1 mild hardening + gate warmup、D2 mild hardening + no internal GRU、D3 mild hardening + gate warmup + no internal GRU 和 D4 medium hardening + gate warmup + no internal GRU 配置
- **AND** D 组配置 MUST 不启用 destructive `csi_degradation`

#### Scenario: destructive negative control 使用已有 degradation
- **WHEN** 用户加载 A2 destructive degradation negative control 配置
- **THEN** 配置 MUST 显式启用 `data.dataset.csi_degradation`
- **AND** 配置 MUST 不启用 `csi_hardening`

### Requirement: Easy modality + CSI 多模态验证配置
系统 MUST 提供 easy modality + CSI 验证配置，用于比较普通 joint training 和 CSI-prioritized warmup 对 slow CSI 贡献恢复的影响。默认 easy modality MUST 为 GPS，配置 MUST 允许后续替换为其它 easy modality。

#### Scenario: GPS+CSI 验证配置可加载
- **WHEN** 用户加载多模态验证配置
- **THEN** 系统 MUST 至少提供 E0 GPS-only、E1 GPS+clean CSI joint、E2 GPS+slow CSI joint 和 E3 GPS+slow CSI + CSI-prioritized warmup 配置
- **AND** E1 到 E3 MUST 使用 `modalities: [gps, csi]` 或等价归一化后的模态集合

#### Scenario: CSI-prioritized warmup 配置表达训练阶段
- **WHEN** 用户加载 E3 CSI-prioritized warmup 配置
- **THEN** 配置 MUST 表达前期只训练或优先训练 CSI encoder 与 fusion/head 的阶段
- **AND** 配置 MUST 表达 warmup 结束后 GPS 与 CSI 联合训练的阶段

### Requirement: CSI hardening sweep 分析脚本
系统 SHOULD 保留 CSI hardening debug/sweep 的解释边界和关键判定阈值，但 MUST 不要求长期维护一次性 `scripts/analyze_csi_hardening_sweep.py` 分析脚本。需要复查 sweep 时，开发者 SHOULD 使用 run history、resolved config、debug matrix parity 和 `docs/research_notes.md` 中的 high-ceiling/slow-to-learn 判定阈值。

#### Scenario: 历史分析脚本可退役
- **WHEN** 当前 workflow 不再需要 `scripts/analyze_csi_hardening_sweep.py`
- **THEN** 项目 MAY 删除该脚本和只服务它的测试
- **AND** CSI hardening 配置、训练脚本和文档 MUST 不继续要求该脚本存在

#### Scenario: 解释边界保留
- **WHEN** 开发者解释 CSI hardening sweep 或 debug matrix
- **THEN** 文档 MUST 保留 destructive negative control、high-ceiling/slow-to-learn、clone parity 和 debug-first caveat
- **AND** 结论 MUST 不把未验证本地 run 写成正式结果 claim

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

### Requirement: CSI hardening sweep rerun workflow
项目 MUST 提供修复后的 CSI-only A/B/C/D sweep 运行入口或命令说明。该 workflow MUST 先运行短 debug gate，再运行完整 CSI-only sweep，并在输出中记录所使用的配置版本、pilot estimation 模式、noise ratio diagnostics 和旧结果隔离状态。

#### Scenario: 生成修复后的 A1 配置
- **WHEN** 开发者生成或加载修复后的 A1 mild pilot estimation 配置
- **THEN** 配置 MUST 使用 estimation-SNR 模式
- **AND** resolved config MUST 记录固定 SNR 或训练 SNR 采样区间

#### Scenario: 生成修复后的 B/C/D 配置
- **WHEN** 开发者生成或加载修复后的 B、C 或 D 组配置
- **THEN** 每个配置 MUST 显式关闭 pilot estimation noise
- **AND** 每个配置 MUST 保留自身声明的 hardening 或 encoder 变量

#### Scenario: 重跑前执行 debug gate
- **WHEN** 开发者请求完整 CSI hardening sweep
- **THEN** workflow MUST 先确认 A0 original、A0 clone、pilot disabled、C1 only 和 C2 only 的 debug gate 通过
- **AND** 如果 gate 未通过，workflow MUST 停止或将完整 sweep 输出标记为 pending-debug

#### Scenario: 输出新旧结果隔离状态
- **WHEN** 修复后的 sweep analysis 完成
- **THEN** summary artifact MUST 记录当前 sweep 是否基于修复后的 pilot scaling 配置
- **AND** 如果同一项目中存在旧 invalid sweep，summary artifact MUST 不把旧 sweep 的候选结果混入当前 ranking
