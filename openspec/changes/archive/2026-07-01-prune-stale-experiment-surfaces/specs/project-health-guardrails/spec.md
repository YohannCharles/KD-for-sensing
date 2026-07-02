## ADDED Requirements

### Requirement: Stale current 引用必须被健康护栏发现
项目健康护栏 MUST 检查 current README、docs、OpenSpec specs、inventory、tests 和 pyproject 中的当前入口引用是否指向真实存在的源码、配置或 console script。已删除入口不得继续作为 current scenario 或推荐命令出现。

#### Scenario: 已删除 BeamBench CLI 引用失败
- **WHEN** current spec 或 docs 要求 `kd_sensing.cli.beambench_check_dataset` 作为当前入口
- **THEN** 架构边界或 OpenSpec 校验任务 MUST 要求改为当前 owner module 或等价保留入口
- **AND** 该旧 CLI 文件 MUST 不因 spec 漂移被恢复

#### Scenario: 历史引用不误报
- **WHEN** archive、历史报告或明确标记为 retired/local/manual 的段落引用已删除入口
- **THEN** 健康护栏 MAY 允许该引用
- **AND** 该段落 MUST 不把旧入口描述为 quickstart、active mainline 或长期推荐 workflow

### Requirement: 本地脚本和配置分类漂移必须被检查
项目健康护栏 MUST 检查 `scripts/`、`configs/scene31/`、local/manual experiment YAML 和诊断 manifest 的 lifecycle 分类与真实文件系统一致。新增或保留的 local/manual surface MUST 有 owner、输出边界、是否推荐、删除触发条件和 focused 验证说明。

#### Scenario: 新 local script 未登记
- **WHEN** `scripts/` 下存在新的 Python 或 shell 文件
- **THEN** 架构边界检查 MUST 要求 inventory、OpenSpec tasks 或 current docs 记录其 lifecycle 和输出边界
- **AND** 未登记脚本 MUST 不作为 current surface 静默通过

#### Scenario: 固定 GPU shell 不作为 package workflow
- **WHEN** 保留脚本只固定 GPU 映射、日志目录和一组本地 YAML
- **THEN** 健康护栏 MUST 要求其分类为 local/manual shell orchestration
- **AND** README quickstart MUST 不把它升级为 package CLI 或长期 workflow

### Requirement: 旧实验 facade 不得回流实现
项目健康护栏 MUST 防止旧实验 facade 或兼容 reader 重新承载大段实现。若 `cnn_hybrid_jepa_visual_prior_sweep`、`jepa_gps_shortcut_benchmark` 或 MMW preparation facade 被保留，它们 MUST 只暴露必要公开入口或委托 owner 模块。

#### Scenario: 兼容 reader 重新变厚
- **WHEN** 旧 full sweep 兼容模块新增训练 runner、job graph、shell generation 或 cleanup 逻辑
- **THEN** 架构边界检查 MUST 失败
- **AND** 修复路径 MUST 是迁回当前 owner、删除旧逻辑或更新 OpenSpec 明确恢复该 workflow

#### Scenario: CLI glue 允许薄委托
- **WHEN** package CLI 只解析参数并调用当前 owner module
- **THEN** 健康护栏 MUST 允许该引用
- **AND** 允许范围 MUST 不扩展到内部 runtime 模块从 facade 导入 helper
