## ADDED Requirements

### Requirement: 退役失败实验路线不得保留源码支持面
当用户明确退役某条失败研究路线并要求不保留兼容时，项目 MUST 从当前源码支持面删除该路线的公开入口、配置、实现模块、测试和文档推荐路径。系统 MUST 不新增兼容 alias、stub CLI、薄 facade 或 registry fallback 来维持旧路线可发现性。

#### Scenario: 退役入口不可安装
- **WHEN** 开发者刷新 editable install 后检查 console scripts
- **THEN** 已退役路线的 `kd-sensing-*` 命令 MUST 不再由 `pyproject.toml` 声明
- **AND** 项目 MUST 不提供等价旧命令 alias 或兼容包装层

#### Scenario: 退役实现不可作为当前模块导入
- **WHEN** 开发者检查 `src/kd_sensing/cli`、`data`、`engine`、`models` 和 `losses`
- **THEN** 已退役路线专属模块 MUST 不再作为当前源码模块保留
- **AND** 保留主线不得从这些退役模块导入 helper

### Requirement: Top8 residual coarse 退役边界
Top8 selector 训练/plot/compare、GPS coarse anchor、GPS prior residual/delta correction 和 camera residual MUST 不属于当前包结构和推荐入口。BGAM、BGAM 依赖的 TopK candidate manifest/loss 支撑代码、通用 Top-K 指标、circular metrics、GPS-Rel-Polar、GPS v2、CSI、Raymobtime、JEPA 和 viewer manifest MAY 保留。

#### Scenario: 保留通用指标
- **WHEN** 清理实现扫描到 `topk`、`candidate` 或 `residual` 字符串
- **THEN** 系统 MUST 按语义判断归属
- **AND** 普通 evaluation Top-K、viewer top-k 展示、CSI candidate ranking 和 GPS v2 自身 residual 诊断不得仅因字符串命中被删除

## REMOVED Requirements

### Requirement: Residual workflow 使用包内 CLI
**Reason**: DeepSense6G residual correction 已作为失败研究路线退役，不再需要包内 CLI 约束。
**Migration**: 无兼容迁移；使用保留的 GPS v2、supervised/adaptation、CSI、Raymobtime、JEPA 或 viewer workflow。

### Requirement: DeepSense6G Top8 selector 包内入口
**Reason**: DeepSense6G Top8 selector 已退役，不再作为当前包内入口维护。
**Migration**: 无兼容迁移。
