## ADDED Requirements

### Requirement: 健康护栏验证 T2/baseline 唯一 surface
架构和配置健康检查 MUST 从 tracked recipes、`pyproject.toml`、inventory 和 active T2 artifacts 验证只有 T2、S1、AMBER-Full、RMBP-MM current surface。检查 MUST 拒绝 tracked runtime artifacts、已删除路线的 current import/reference 和任何从 `outputs/` 读取 canonical recipe 的 launcher。

#### Scenario: 删除批次可快速验收
- **WHEN** implementation 删除非 T2/baseline source、CLI、script、config 或测试
- **THEN** focused validation MUST 至少覆盖 config load、T2/baseline import/forward、public CLI help 和 stale reference scan
- **AND** 所有项目 Python 验证 MUST 使用 `conda run -n kd_mm_beam`
