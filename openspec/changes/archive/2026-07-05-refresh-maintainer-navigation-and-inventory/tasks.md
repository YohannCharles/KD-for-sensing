## 1. 导航和 inventory 同步

- [x] 1.1 在 `docs/agent_navigation.md` 顶部增加当前项目一屏摘要，覆盖当前主线、入口、退役边界、必读文件和快速验证命令。
- [x] 1.2 刷新 `docs/project_surface_inventory.md` 的 tracked 规模基线和 `configs/` YAML 数量说明，保留“趋势信号而非硬 KPI”的解释。
- [x] 1.3 在 `docs/project_surface_inventory.md` 中登记 4 个 Scene31 / Scene31-34 报告脚本的 lifecycle、职责和 ignored 输出边界。

## 2. 验证和收口

- [x] 2.1 运行 `openspec validate refresh-maintainer-navigation-and-inventory --strict`。
- [x] 2.2 运行 `openspec validate --all --strict`。
- [x] 2.3 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [x] 2.4 运行 `conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_config_load_characterization.py -q`。
