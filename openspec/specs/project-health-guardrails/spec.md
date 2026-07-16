# project-health-guardrails Specification

## Purpose

定义验证 T2/baseline 唯一 surface 的最小健康护栏，使删除批次可以通过不写入运行状态的 config、import、CLI 与 stale-reference 检查快速验收。

## Requirements

### Requirement: 健康护栏验证 T2/baseline 唯一 surface

架构、config 与 CLI 检查 MUST 从 tracked recipes、`pyproject.toml`、inventory 与 active T2 artifacts 验证 current surface 只包含 T2、S1、AMBER-Full、RMBP-MM。

#### Scenario: 删除批次验收

- **WHEN** implementation 删除非 retained source、CLI、script、config 或测试
- **THEN** focused validation MUST 覆盖 config load、import/forward、public CLI help 与 stale reference scan
- **AND** 所有 Python 验证 MUST 使用 `conda run -n kd_mm_beam`

### Requirement: 护栏不修改运行状态

健康检查 MUST 不读取真实 dataset、不启动训练且不写入 output、logs、cache 或 checkpoint。

#### Scenario: 运行快速健康检查

- **WHEN** 维护者执行 architecture、config 或 CLI focused validation
- **THEN** 检查 MUST 只读取 tracked source 与测试 fixture
- **AND** 不得创建或修改本地运行产物
