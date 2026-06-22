## ADDED Requirements

### Requirement: 低价值 facade 和 thin wrapper 必须收缩
项目 MUST 删除或最小化只 re-export owner 符号、只转发 CLI `main`、只维护旧 import path 或只镜像 `__all__` 的低价值 facade。保留 facade 时，facade MUST 是当前明确公开 API，且 MUST 不导入重依赖、不注册默认组件、不承载旧 alias 和长期实现逻辑。

#### Scenario: 删除 3 行 CLI wrapper
- **WHEN** 某个包内 CLI 文件只导入另一个模块的 `main` 并导出 `__all__`
- **THEN** `pyproject.toml` 的 console script MUST 直接指向真实 owner `main`
- **AND** 删除 wrapper 后对应 `kd-sensing-* --help` MUST 继续可运行

#### Scenario: 内部代码不用 package facade
- **WHEN** 内部源码或测试需要某个实现符号
- **THEN** 它 MUST 从真实 owner module 导入
- **AND** 它 MUST 不通过 package-level re-export、旧 alias facade 或 aggregate module 维持旧路径

### Requirement: 内部 `__all__` 镜像不得成为维护负担
内部模块 MUST 不为了镜像所有可见符号而维护大型 `__all__` 表。`__all__` 只允许用于稳定 public facade、明确 plugin/export 边界或避免 wildcard import 的必要模块。

#### Scenario: 删除无用 `__all__`
- **WHEN** 某个模块没有 current docs 推荐 wildcard import，也不是稳定 public facade
- **THEN** 本 change MAY 删除该模块的 `__all__`
- **AND** 显式 import 调用方 MUST 继续工作

### Requirement: 退役整模型类不作为包结构保留对象
已由 `modular_sequence` 或当前 whole-model exception 替代、且从 registry 移除的旧整模型类 MUST 不再作为直接导入 public API 保活。仍被当前路径使用的 feature extractor 或子组件 MUST 保留在 owner module。

#### Scenario: 删除旧整模型类保留特征提取器
- **WHEN** 旧 strong/lightweight modality model 已退出 registry 但同文件 feature extractor 仍被当前模块化模型使用
- **THEN** 实现 MUST 删除退役整模型类和 alias
- **AND** 实现 MUST 保留 feature extractor 并保持当前 `modular_sequence` 构建可用
