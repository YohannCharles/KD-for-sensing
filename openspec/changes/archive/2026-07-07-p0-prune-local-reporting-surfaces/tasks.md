## 1. Baseline 与候选确认

- [x] 1.1 运行 `git status --short`，确认本 change 不包含本地数据、outputs、logs、cache、checkpoint 或历史权重。
- [x] 1.2 枚举 `scripts/analysis/*.py`、Scene31-34 final analysis cluster、Scene31 summary cluster 和 `scripts/reevaluate_apples_to_apples.py` 的 current docs/spec/tests 引用。
- [x] 1.3 为每个候选标注 `delete`、`consolidate` 或 `retained-with-reason`，并记录替代 owner。

## 2. scripts/analysis 一次性脚本退出 current surface

- [x] 2.1 确认每个 `scripts/analysis/*.py` 的结论、输入 artifact 和关键输出是否已沉淀到 docs、paper tables、claim notes 或 retained artifact 说明。
- [x] 2.2 删除已沉淀的一次性脚本；若仍需保留，补充 retained-with-reason 和 focused validation。
- [x] 2.3 更新 README/docs/OpenSpec references，避免 current workflow 推荐已删除路径。

## 3. Scene31-34 final analysis 收敛

- [x] 3.1 设计一个 Scene31-34 final analysis owner，覆盖 profile、significance、paper table、conclusion、CDF、heatmap、sampling 和 presentation artifact 的当前输出契约。
- [x] 3.2 将重复脚本中的共享读取、筛选、格式化和输出逻辑迁到 owner 或窄 helper。
- [x] 3.3 删除被 owner 覆盖的 per-artifact 脚本，不新增同职责 wrapper。
- [x] 3.4 对纸面表格、结论文本和关键图表输出做字段级或 snapshot 对照。

## 4. Scene31 summary 收敛

- [x] 4.1 建立参数化 Scene31 summary owner，覆盖 BC-next、P0 fresh eval、baseline pack、subset reliability、patternfilm、funnel、next-round 和 subset reference。
- [x] 4.2 将各 workflow 的 profile/group、默认路径、输出字段和错误信息显式登记。
- [x] 4.3 删除重复 summary 脚本；docs/tests/inventory 指向 consolidated owner。
- [x] 4.4 运行相关 Scene31 focused tests 或等价 smoke checks。

## 5. apples-to-apples 复评归位

- [x] 5.1 对比 `scripts/reevaluate_apples_to_apples.py` 与 package evaluate workflow 的职责重叠。
- [x] 5.2 将仍需要的 fresh evaluation 行为折入 canonical eval owner 或文档化 recipe。
- [x] 5.3 删除独立大型复评脚本，或记录 retained-with-reason 与删除触发条件。

## 6. 防回流与验证

- [x] 6.1 更新 `docs/project_surface_inventory.md`、architecture boundary 或 scripts surface doctor，覆盖删除项和替代 owner。
- [x] 6.2 运行 `openspec validate p0-prune-local-reporting-surfaces --strict`。
- [x] 6.3 运行 `openspec validate --all --strict`。
- [x] 6.4 运行 architecture、CLI/config、Scene31/Scene31-34 focused validation。
- [x] 6.5 最终说明列出删除行数、保留理由、未运行验证和后续 archive 建议。
