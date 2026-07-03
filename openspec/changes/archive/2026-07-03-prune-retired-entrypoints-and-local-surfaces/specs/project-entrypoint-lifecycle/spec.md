## ADDED Requirements

### Requirement: Module-only CLI must be public or deleted
`src/kd_sensing/cli/*.py` 中可直接运行的 module-only CLI MUST 要么在 `pyproject.toml` 声明为 `kd-sensing-*` console script 并出现在 README/current docs 或 current spec 中，要么从当前支持面删除。Shared helper 例如 `cli/common.py` MAY 保留，但 MUST 不提供独立 `main()` 入口或用户可见 workflow。

#### Scenario: Hidden CLI cleanup
- **WHEN** 一个 `kd_sensing.cli.<name>` 模块包含 `main()` 或 console-style parser
- **THEN** 项目 MUST 在 `pyproject.toml` 声明对应 console script，或删除该 CLI wrapper
- **AND** current docs MUST 不推荐未声明 console script 的隐藏 `python -m kd_sensing.cli.<name>` 入口

#### Scenario: Shared CLI helper
- **WHEN** 一个 CLI 模块只提供配置加载、argparse helper 或 shared exit handling
- **THEN** 它 MAY 保留为 internal helper
- **AND** 架构边界测试 MUST 不把它当成 public runnable entrypoint

### Requirement: Local/manual scripts are removable unless explicitly retained
`scripts/` 下本地研究、固定 GPU queue、one-shot 分析和 shell orchestration MUST 具备 current lifecycle、owner、输出边界和删除条件；没有 current docs/spec/result registry 引用、没有替代价值或已有 package CLI 覆盖的脚本 MUST 删除。

#### Scenario: Script has no current owner
- **WHEN** tracked `scripts/*.py`、`scripts/**/*.py` 或 `scripts/*.sh` 不属于 dataset preparation、config generator、current research diagnostic 或 explicitly retained local/manual runner
- **THEN** 该脚本 MUST 从源码表面删除
- **AND** README、docs 和 OpenSpec MUST 不把它描述为当前入口

#### Scenario: Script duplicates package CLI
- **WHEN** 一个脚本只包装 `kd-sensing-train`、`kd-sensing-evaluate`、`kd-sensing-preprocess` 或其它 package console script
- **THEN** 该脚本 MUST 删除
- **AND** 用户文档 MUST 指向 package console script
