## ADDED Requirements

### Requirement: 项目健康护栏纳入架构边界
项目 MUST 将健康护栏纳入架构边界测试，使包结构、轻量导入、入口 allowlist、热点 inventory、测试 bootstrap 和分层验证命令保持一致。架构边界测试 MUST 能在全量 pytest 之前暴露项目支持面或维护性边界漂移。

#### Scenario: 架构边界检查健康护栏文件
- **WHEN** 开发者运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
- **THEN** 测试 MUST 验证 shared pytest bootstrap、项目表面积 inventory 和当前健康检查命令说明存在且互相一致
- **AND** 测试 MUST 拒绝新增未分类的长期脚本入口、未登记热点或明显回流的兼容 facade helper

#### Scenario: 新增热点必须声明边界
- **WHEN** 新代码引入超长 orchestration 函数、dataset 类、manifest builder 或兼容 facade
- **THEN** 变更 MUST 同步更新热点 inventory 或拆分到职责明确的窄模块
- **AND** 架构边界测试 MUST 提供可定位到文件和符号的失败信息

### Requirement: 测试基础设施不得重复项目路径注入
项目 MUST 将普通测试的源码路径注入集中管理。除架构边界 import probe、subprocess smoke 或明确隔离环境测试外，测试文件不得各自维护重复的 `ROOT/SRC/sys.path.insert` 启动逻辑。

#### Scenario: 普通测试文件不复制 bootstrap
- **WHEN** 开发者新增或修改普通单元测试
- **THEN** 测试 MUST 通过 shared pytest bootstrap 导入 `kd_sensing`
- **AND** 文件级 `sys.path.insert` 复制片段 MUST 被架构边界测试拒绝或要求显式例外说明

#### Scenario: 子进程边界测试可控
- **WHEN** 架构边界测试需要在干净解释器中验证某个 import 不牵出重依赖
- **THEN** 该测试 MAY 在子进程代码中显式设置 `sys.path`
- **AND** 该例外 MUST 保持局部，不得作为普通测试模板传播
