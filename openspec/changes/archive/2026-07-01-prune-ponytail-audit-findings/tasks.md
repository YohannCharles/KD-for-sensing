## 1. OpenSpec 状态收口

- [x] 1.1 运行 `openspec list --json` 和 `openspec status --change add-rbma-prototype-kd-missing-workflow`，确认已完成 active change 是否仍未归档
- [x] 1.2 若无阻塞，归档 `add-rbma-prototype-kd-missing-workflow`；若暂不归档，在本 change artifact 或最终说明中记录 deferral 原因和影响范围
- [x] 1.3 修复 current specs 中的归档 `TBD - created by archiving` Purpose，至少覆盖当前扫描发现的 TII VLRG、reused-weight fusion diagnostics、AMBER-lite、AMR-Net、WCL2025 missing-modality 和 GPS-query evidence specs
- [x] 1.4 重新检查 lifecycle inventory，确认保留的 retired tombstone 仍有 migration guard 价值，无法证明价值的只列入后续折叠候选

## 2. 配置和脚本表面分类

- [x] 2.1 扫描 `configs/fusion/*.yaml`，对比 `docs/project_surface_inventory.md` 的 root 保留分类，列出未登记 root YAML
- [x] 2.2 将非 root canonical/current thin entry 的实验 YAML 迁入 `configs/fusion/experiments/<family>/`、登记为 local/manual，或删除无 current 引用的临时配置
- [x] 2.3 审核新增 Scene31/M2Beam/RBMA fullrun/strong-encoder/queue YAML，记录 owner、输入输出边界、claim caveat 和删除触发条件
- [x] 2.4 扫描 `scripts/` 下新增 Python/shell 文件，将其分类为 research diagnostic、dataset preparation、figure helper、shell orchestration 或 local/manual artifact
- [x] 2.5 对根目录 `test.md` 或同类 runbook 执行分类：迁移长期有价值内容、标记为历史/本地笔记，或删除无 current 价值的 tracked 临时文件
- [x] 2.6 更新 README/docs/OpenSpec/tests 中受迁移配置或脚本影响的路径引用，确保 current 引用都指向真实存在的文件

## 3. 低风险源码表面修复

- [x] 3.1 将 `src/kd_sensing/diagnostics/jepa_visual_analysis.py` 中对 `kd_sensing.diagnostics.jepa_gps_shortcut_benchmark` facade 的内部导入改为直接导入 `jepa_benchmark_common.py`、`jepa_benchmark_runner.py` 或对应 owner 模块
- [x] 3.2 保留 `jepa_gps_shortcut_benchmark.py` 作为公开 facade，不删除 CLI 兼容入口
- [x] 3.3 将 `.codegraph/daemon.pid` 从 git 跟踪中移除，并更新 `.codegraph/.gitignore` 覆盖 pid、socket、数据库、WAL、cache 和 log 等本地 CodeGraph 状态
- [x] 3.4 扫描普通 pytest 文件中的文件级 `ROOT/SRC/sys.path.insert` 片段，删除非 subprocess/import-boundary probe 所需的重复 bootstrap
- [x] 3.5 仅在引用清晰且改动很小的情况下合并 U-Mask eval/export 重复 helper；若会触碰评估语义，改为登记后续 change

## 4. Guardrail 和 Inventory 更新

- [x] 4.1 更新 `docs/project_surface_inventory.md`，同步真实 root fusion YAML、experiment config family、脚本 lifecycle、CodeGraph 本地状态边界和 facade 回流策略
- [x] 4.2 更新 `docs/agent_navigation.md` 或相关导航说明，记录完成 active change、未分类表面和本地工具状态的读取顺序
- [x] 4.3 扩展 `tests/test_architecture_boundaries.py`，拒绝 current specs 中的 `TBD` Purpose 或归档脚手架
- [x] 4.4 扩展架构边界测试，验证 `configs/fusion/*.yaml` 与 inventory root 分类一致
- [x] 4.5 扩展架构边界测试，发现未分类 `scripts/` 文件、root runbook 或临时 queue 脚本时失败
- [x] 4.6 扩展 facade 回流检查，允许 package CLI 和 facade 文件本身，拒绝 diagnostics/engine/data/models/losses/evaluation 内部从 facade 导入窄 helper
- [x] 4.7 扩展 tracked runtime artifact 检查，覆盖 `.codegraph/daemon.pid` 和其它本地工具状态
- [x] 4.8 扩展普通 pytest bootstrap 检查，保留 subprocess/import-boundary probe 例外

## 5. 验证

- [x] 5.1 运行 `openspec validate prune-ponytail-audit-findings --strict`
- [x] 5.2 运行 `openspec validate --all --strict`
- [x] 5.3 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
- [x] 5.4 运行 `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py tests/test_cli_help.py -q`
- [x] 5.5 若修改 U-Mask eval/export helper，运行 `conda run -n kd_mm_beam pytest tests/test_u_mask_beam_jepa.py tests/test_u_mask_beam_jepa_eval_matrix.py -q`
- [x] 5.6 若修改脚本或 CLI glue，运行对应无副作用 `--help` 或 dry-run smoke；不得运行已退役 viewer manifest / visualize-modalities CLI
- [x] 5.7 运行 `git status --short`，确认变更不包含 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、TensorBoard event 或 `All_models/` 新产物
- [x] 5.8 记录未运行验证的原因、环境限制和剩余风险
