## ADDED Requirements

### Requirement: DeepSense6G Top8 selector 包内入口
项目 MUST 将 DeepSense6G GPS Top8 Candidate Selector 的实现放入 `src/kd_sensing/` 包内。manifest、dataset、model、loss、engine、plotter 和 comparison CLI MUST 按现有职责边界分布在 `kd_sensing.data`、`kd_sensing.models`、`kd_sensing.losses`、`kd_sensing.engine`、`kd_sensing.evaluation`、`kd_sensing.cli` 或 `kd_sensing.utils` 中。项目 MUST NOT 新增长期维护的顶层 `src.data.*`、`src.models.*`、`src.losses.*` 或 `src.run_*.py` 运行入口。

#### Scenario: console scripts 暴露 Top8 selector workflow
- **WHEN** 开发者完成 editable install 并查看 `pyproject.toml` entry points
- **THEN** 项目 MUST 暴露 Top8 selector 相关 console scripts
- **AND** scripts MUST 至少覆盖 manifest 构建、selector 运行、plotter 和 GPS v2 comparison
- **AND** 每个 console script MUST 委托 `kd_sensing.cli.*` 中的包内实现

#### Scenario: 包内 module CLI 可运行
- **WHEN** 用户执行 `conda run -n kd_mm_beam python -m kd_sensing.cli.prepare_deepsense6g_top8_candidate_manifest --help`
- **THEN** 命令 MUST 正常退出
- **AND** 帮助信息 MUST 包含 `--config`、`--support-ratio`、`--label-space` 和 `--topk`

#### Scenario: 不新增绕过包结构的 src 入口
- **WHEN** 架构边界测试扫描新 workflow
- **THEN** 测试 MUST 验证内部代码不依赖顶层 `src.data`、`src.models`、`src.losses` 或 `src.run_deepsense6g_top8_selector`
- **AND** 用户文档 MUST 推荐 `kd-sensing-*` 或 `python -m kd_sensing.cli.*` 命令

#### Scenario: 轻量导入边界保持稳定
- **WHEN** 开发者执行 `import kd_sensing` 或导入配置/路径轻量模块
- **THEN** 系统 MUST 不因 Top8 selector workflow eager import torch dataset、matplotlib plotter、pandas manifest builder 或训练 runtime
- **AND** Top8 selector 重依赖模块 MUST 只在对应 CLI、engine 或显式模块导入时加载
