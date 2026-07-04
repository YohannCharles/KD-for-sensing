## ADDED Requirements

### Requirement: 大测试拆分后健康护栏覆盖不得下降
Architecture 和 focused tests MAY 拆分为更小文件，但 MUST 继续覆盖 retired route 回流、tracked runtime artifact、current path/config 引用、facade 回流和 script lifecycle 检查。

#### Scenario: architecture boundary 拆分后仍拒绝关键回归
- **WHEN** tests are reorganized
- **THEN** `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` or its documented replacement MUST still fail on old entrypoint回流, tracked output artifacts and invalid current references
- **AND** documentation MUST point to the replacement command if the file is split
