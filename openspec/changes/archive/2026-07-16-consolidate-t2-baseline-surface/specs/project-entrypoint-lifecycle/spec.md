## MODIFIED Requirements

### Requirement: Public CLI 仅保留训练、评估和预处理

项目 MUST 只声明 `kd-sensing-train`、`kd-sensing-evaluate` 和 `kd-sensing-preprocess` 三个 public console script。T2/baseline 的多 seed orchestration、fixed-mask evaluation 与 summary MUST 作为 inventory 中有 owner 的 local/manual script 保留，其他 CLI MUST 从 public surface 删除。

#### Scenario: 安装后只暴露最小 public CLI

- **WHEN** 用户在 `kd_mm_beam` 环境中安装项目并检查 `pyproject.toml`
- **THEN** entry points MUST 只包含 train、evaluate 和 preprocess
- **AND** CLI help tests MUST 只验证这三个命令

#### Scenario: T2 local/manual scripts 有生命周期

- **WHEN** tracked `scripts/` 文件服务 T2/baseline matrix、BPA/CMA ablation 或 hyperparameter screening
- **THEN** inventory MUST 记录其 owner 和输出边界
- **AND** 脚本 MUST 不被包装为额外 public console script
