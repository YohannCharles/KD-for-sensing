## ADDED Requirements

### Requirement: Console script surface guardrail
项目健康护栏 MUST 检查 `pyproject.toml` console scripts、CLI help smoke、project surface inventory 和 current docs/OpenSpec 的一致性。新增、删除或降级 public CLI 时，guardrail MUST 能发现缺少生命周期分类、缺少 smoke、docs stale 引用或已删除命令回流。

#### Scenario: pyproject 与 help smoke 一致
- **WHEN** 开发者运行 CLI/architecture 健康检查
- **THEN** 检查 MUST 比对 `pyproject.toml` 中的 `kd-sensing-*` entry points 和保留 public CLI 的 help smoke 覆盖
- **AND** 缺少 smoke 的 public CLI MUST 被报告，除非 OpenSpec 明确将其标记为不需要 help smoke 的例外

#### Scenario: docs 不引用已删除 public CLI
- **WHEN** public console script 被删除或降级为 internal-only
- **THEN** README、docs、OpenSpec current specs 和 tests MUST 不再把旧命令描述为 current public entrypoint
- **AND** 健康检查 MUST 能发现 current 文档中的 stale command reference

#### Scenario: 新 public CLI 需要生命周期锚点
- **WHEN** 后续 change 新增 `kd-sensing-*` console script
- **THEN** architecture/surface 检查 MUST 要求同步 owner module、inventory/docs 引用、help smoke 和输出边界
- **AND** 缺少这些锚点时检查 MUST 失败或在 doctor 中报告 error

### Requirement: Public entrypoint cleanup validation
公共入口瘦身完成后，项目 MUST 运行分层验证，覆盖 OpenSpec、console script help、architecture boundary 和 surface doctor。验证 MUST 不启动真实训练、不读取真实数据、不写入 checkpoint 或 runtime outputs。

#### Scenario: right-size CLI 验收
- **WHEN** public entrypoint cleanup implementation 完成
- **THEN** 开发者 MUST 运行 `openspec validate right-size-public-entrypoint-surface --strict`
- **AND** 开发者 MUST 运行 `openspec validate --all --strict`
- **AND** 开发者 MUST 运行 `conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_architecture_boundaries.py -q`

#### Scenario: doctor 验收
- **WHEN** implementation 修改 pyproject、CLI module、inventory 或 current docs 中的 public entrypoint
- **THEN** 开发者 MUST 运行 project surface doctor 的 scripts/configs/hotspots 或包含 CLI surface 的等价 scope
- **AND** 未运行的验证 MUST 在最终说明中记录原因和剩余风险
