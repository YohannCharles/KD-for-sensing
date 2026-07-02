## ADDED Requirements

### Requirement: Scene31 BTAPA local ablation workflow
项目 MUST 提供 Scene31 BTAPA local/manual ablation 配置、串行 launcher 和只读分析脚本。该 workflow MUST 使用当前 `kd-sensing-train` CLI，不得新增旧 root 训练入口。

#### Scenario: BTAPA launcher dry-run
- **WHEN** 用户运行 `conda run -n kd_mm_beam bash scripts/run_btapa_experiments.sh --dry_run --num_workers 4 --max_parallel 1`
- **THEN** launcher MUST 只打印每个 BTAPA 实验的训练命令
- **AND** 默认 `max_parallel` MUST 为 1

#### Scenario: BTAPA 输出隔离
- **WHEN** 用户运行任一 BTAPA 配置
- **THEN** 输出 run name MUST 包含 `btapa`
- **AND** 系统 MUST 不覆盖旧 `main_v3_strong_reliability_proto` 输出
