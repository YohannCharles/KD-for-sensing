## ADDED Requirements

### Requirement: 可复制 verify 入口
项目 MUST 提供或记录一个可复制的最小 verify 入口，用于聚合 OpenSpec strict、架构边界、CLI help、配置 characterization 和无数据 synthetic smoke。该入口 MUST 不启动真实训练、不读取真实 `dataset/`、不加载真实 checkpoint、不写入训练产物。

#### Scenario: 运行 quick verify
- **WHEN** 开发者运行项目记录的 quick verify 命令
- **THEN** 命令 MUST 覆盖 `openspec validate --all --strict`
- **AND** 命令 MUST 覆盖 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
- **AND** 命令 MUST 不读取真实本地数据或写入 checkpoint

#### Scenario: CLI/config verify 分层
- **WHEN** 变更触碰 console script、CLI parser、config loader 或 virtual config
- **THEN** 项目 MUST 记录可复制的 CLI/config verify 命令
- **AND** 该命令 MUST 使用 `conda run -n kd_mm_beam`

### Requirement: 最小环境声明
项目 MUST 提供 tracked 的最小环境声明或环境生成说明，用于重建无数据 smoke 验证环境。环境声明 MUST 区分 smoke/dev 依赖与 GPU 训练依赖，并 MUST 不包含本地数据路径、密码、token 或 checkpoint。

#### Scenario: 新机器重建 smoke 环境
- **WHEN** 维护者在新机器上准备运行无数据健康检查
- **THEN** 文档或环境文件 MUST 给出安装项目和运行 quick verify 的最小步骤
- **AND** 步骤 MUST 不要求真实 `dataset/`、`outputs/` 或 `All_models/` 可用

### Requirement: 轻量 lint 和脚本编译检查
项目 MUST 提供轻量 lint 或 compile 检查，用于在全量 pytest 前发现 Python 语法错误、脚本入口错误和明显文档引用漂移。该检查 MUST 不替代 focused tests。

#### Scenario: 脚本语法错误快速暴露
- **WHEN** tracked `scripts/` 或 package CLI 文件存在 Python 语法错误
- **THEN** 轻量检查 MUST 在真实训练前失败
- **AND** 失败信息 MUST 指向具体文件
