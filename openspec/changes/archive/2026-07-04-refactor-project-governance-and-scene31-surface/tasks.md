## 1. Baseline 与治理对齐

- [x] 1.1 扫描 tracked `configs/scene31/`、`scripts/run_scene31_*.sh`、Scene31 generator/summary/checkpoint selection 工具，记录当前数量、owner、默认输出 root 和引用测试。
- [x] 1.2 更新 `docs/project_surface_inventory.md` 中的 Scene31 lifecycle、源码/YAML/script 统计和 local/manual caveat。
- [x] 1.3 更新 `docs/agent_navigation.md`，加入 Scene31 workflow 修改前的 inventory/OpenSpec/验证路由。
- [x] 1.4 若相关 completed change 仍未归档，运行或记录 `openspec archive <change>` 的处理决定。

## 2. Scene31 表面积收敛

- [x] 2.1 判定 `configs/scene31/` 中哪些实体 YAML 必须保留，哪些可由 generator + manifest 本地重建。
- [x] 2.2 收敛 Scene31 generator 的共用 manifest/template 写出逻辑，保持 run name、seed、epoch、sampler、loss 和 output root 字段不变。
- [x] 2.3 抽取 Scene31 runner 中重复的 GPU worker、skip/overwrite、train/eval、failed list 和 summary 调用逻辑。
- [x] 2.4 保持现有 shell/local manual 入口可运行，或在 inventory 中记录删除/替代路径。

## 3. 测试与护栏

- [x] 3.1 更新 `tests/test_scene31_next_round.py`，覆盖 generator 字段、runner safety、summary 输出和必要保留 YAML。
- [x] 3.2 更新 `tests/test_architecture_boundaries.py`，拒绝未登记 Scene31 YAML、runner 或 shell 回流。
- [x] 3.3 运行 `openspec validate refactor-project-governance-and-scene31-surface --strict`。
- [x] 3.4 运行 `conda run -n kd_mm_beam pytest tests/test_scene31_next_round.py tests/test_architecture_boundaries.py -q`。
