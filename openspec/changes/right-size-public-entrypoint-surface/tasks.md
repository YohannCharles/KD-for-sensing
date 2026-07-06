## 1. Baseline 与入口分类

- [ ] 1.1 枚举 `pyproject.toml` 中所有 `[project.scripts]` 的 `kd-sensing-*` entry points，并与 `tests/test_cli_help.py` 当前 help smoke 覆盖做差异表。
- [ ] 1.2 对每个 console script 标记 lifecycle：`core_workflow`、`current_diagnostic`、`paper_export`、`baseline_reproduction`、`local_manual`、`internal_only` 或 `delete`。
- [ ] 1.3 检查每个入口在 README、docs、OpenSpec current specs、tests 和 `docs/project_surface_inventory.md` 中的引用，记录 owner module、职责、输出边界和真实数据/训练副作用边界。
- [ ] 1.4 列出缺少 help smoke、缺少 docs/inventory 锚点、只服务 local/manual、或已有更清晰 owner/public CLI 覆盖的候选入口。

## 2. Public CLI 收缩

- [ ] 2.1 保留 `core_workflow` 和明确 current diagnostic/paper/baseline reproduction 入口，并为缺少 smoke 的保留 CLI 补齐 `--help` 或无副作用 smoke。
- [ ] 2.2 对 `internal_only` 候选删除 `pyproject.toml` console script 声明，保留必要 owner module 或 CLI helper；current docs 不得推荐隐藏 module-only CLI。
- [ ] 2.3 对 `delete` 候选删除 pyproject entry point 和只服务该入口的 CLI wrapper；不得新增 alias、compat wrapper 或 deprecation trampoline。
- [ ] 2.4 确认 `src/kd_sensing/cli/*.py` 中 shared helper 不被误判为 public runnable entrypoint；含 `main()` 的 module-only CLI 必须 public 化、降级或删除。
- [ ] 2.5 保持 CLI glue thin；若 public CLI 文件承载 workflow 主逻辑，将实现迁回 owner module 或记录后续专门 change，不在 CLI 中继续扩张。

## 3. 文档与 Guardrail 同步

- [ ] 3.1 更新 `docs/project_surface_inventory.md`，为所有保留 public CLI 记录 lifecycle、owner、职责、输出边界和 focused validation。
- [ ] 3.2 更新 README、docs、OpenSpec current references，移除已删除或降级 public CLI 的 current 推荐命令，并指向替代入口。
- [ ] 3.3 更新 `tests/test_cli_help.py`，使保留 public CLI 的 help smoke 覆盖完整且不要求已删除命令存在。
- [ ] 3.4 更新 `tests/test_architecture_boundaries.py` 或 surface doctor，使 pyproject、help smoke、inventory/docs 的 entrypoint 漂移会被发现。

## 4. 验证

- [ ] 4.1 运行 `openspec validate right-size-public-entrypoint-surface --strict`。
- [ ] 4.2 运行 `openspec validate --all --strict`。
- [ ] 4.3 运行 `conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_architecture_boundaries.py -q`。
- [ ] 4.4 运行 `conda run -n kd_mm_beam python -m kd_sensing.cli.project_surface_doctor --scope scripts --scope configs --scope hotspots --format markdown --fail-on warning`，或运行实现后支持 CLI surface 的等价 doctor scope。
- [ ] 4.5 对被保留但新增/修改 smoke 的 public CLI 逐个运行 `conda run -n kd_mm_beam <command> --help` 或对应无副作用 dry-run。
- [ ] 4.6 最终说明列出保留、降级、删除的 console scripts，未运行验证原因，以及与 `shrink-experiment-config-families` 并行实现时的 inventory 合并注意事项。
