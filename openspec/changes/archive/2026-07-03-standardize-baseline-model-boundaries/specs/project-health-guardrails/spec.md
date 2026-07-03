## ADDED Requirements

### Requirement: Baseline 与模型目录边界护栏
架构边界测试 MUST 验证 `src/kd_sensing/models/` 和 `src/kd_sensing/baselines/` 的依赖方向。`baselines/` workflow 模块 MUST 不注册 registry-backed 模型或模型子组件；`models/` 模块 MUST 不依赖 `kd_sensing.baselines` workflow package。轻量 package marker 和维护文档 MUST 与该边界一致。

#### Scenario: Baseline workflow 不注册模型组件
- **WHEN** 开发者在 `src/kd_sensing/baselines/` 下新增或修改 Python 文件
- **THEN** 架构边界测试 MUST 拒绝 `@MODELS.register`、`@ENCODERS.register`、`@PROJECTORS.register`、`@REPRESENTATION_CORES.register` 或 `@HEADS.register`
- **AND** 新模型能力 MUST 改放到 `src/kd_sensing/models/` 或通过配置复用已有组件

#### Scenario: 模型实现不反向依赖 workflow
- **WHEN** 开发者在 `src/kd_sensing/models/` 下新增或修改 Python 文件
- **THEN** 架构边界测试 MUST 拒绝从 `kd_sensing.baselines` 导入 workflow 实现
- **AND** 共享模型组件 MUST 通过 `models/`、`engine/`、`data/` 或其它真实 owner 模块复用

#### Scenario: 文档和 marker 不误导维护者
- **WHEN** 维护者查看 baseline package marker、模型目录或项目表面积 inventory
- **THEN** 文档 MUST 将 `baselines/` 描述为 workflow/paper reproduction owner
- **AND** 文档 MUST 不将 `baselines/` 描述为所有 baseline 模型的统一容器
