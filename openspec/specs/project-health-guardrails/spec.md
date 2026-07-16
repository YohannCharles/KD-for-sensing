# project-health-guardrails Specification

## Purpose

定义验证 MMW T2/baseline 与受限 DeepSense6G T2 current surface 的最小健康护栏，使删除批次可以通过不写入运行状态的 config、import、CLI 与 stale-reference 检查快速验收。

## Requirements

### Requirement: 健康护栏验证受限 current surface

架构、config 与 CLI 检查 MUST 从 tracked recipes、`pyproject.toml`、inventory 与 active T2 artifacts 验证 current surface 只包含 MMW T2、S1、AMBER-Full、RMBP-MM 以及 DeepSense6G Scene31–34 T2。检查 MUST 拒绝 tracked runtime artifacts、已删除路线的 current import/reference 和任何从 `outputs/` 读取 canonical recipe 的 launcher。

#### Scenario: 删除批次验收

- **WHEN** implementation 删除非 current source、CLI、script、config 或测试
- **THEN** focused validation MUST 至少覆盖 config load、T2/baseline import/forward、public CLI help 和 stale reference scan
- **AND** 所有 Python 验证 MUST 使用 `conda run -n kd_mm_beam`

### Requirement: 护栏不修改运行状态

健康检查 MUST 不读取真实 dataset、不启动训练且不写入 output、logs、cache 或 checkpoint。

#### Scenario: 运行快速健康检查

- **WHEN** 维护者执行 architecture、config 或 CLI focused validation
- **THEN** 检查 MUST 只读取 tracked source 与测试 fixture
- **AND** 不得创建或修改本地运行产物
