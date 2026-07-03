## 1. Paper Export

- [x] 1.1 定义 paper export 输入 schema、row normalizer 和 output manifest。
- [x] 1.2 实现 claim registry / ledger / summary 到 Markdown、CSV、LaTeX 表格草稿的导出。
- [x] 1.3 实现 stress curve 和 pattern heatmap 的 figure-data CSV/JSON 导出。
- [x] 1.4 添加状态过滤：pending/mock/historical/upper-bound 默认不进入 main table。

## 2. Dataset / Reproducibility Audit

- [x] 2.1 实现只读 dataset audit module/CLI，支持 DeepSense6G、BeamBench 和 MMW layout。
- [x] 2.2 检查 CSV 字段、模态文件引用、label range、beam shift、scene/sample/sequence/timestamp 和 split leakage metadata。
- [x] 2.3 输出 official blocked / local substitute readiness JSON/Markdown report。
- [x] 2.4 更新 `README_REPRODUCE.md` 中不存在脚本的命令，改为 current audit entrypoint 或历史标记。

## 3. Literature Matrix

- [x] 3.1 新增 `docs/literature_matrix.md`，记录 paper id、方法、模态、缺失协议、数据集、指标、官方 artifact、repo relation 和 caveat。
- [x] 3.2 可选新增或更新 `paper/references.bib`，并让 literature matrix 能引用 BibTeX key。
- [x] 3.3 覆盖 DeepSense6G/BeamBench、Vision-Position、AMBER、AMR-Net、RMBP-MM/WCL、JEPA/GPS shortcut 等当前主线相关文献。

## 4. 文档与验证

- [x] 4.1 更新 README、`docs/experiment_matrix.md`、`docs/project_surface_inventory.md` 或相关复现文档索引。
- [x] 4.2 运行 `openspec validate add-paper-export-dataset-audit-literature --strict`。
- [x] 4.3 运行 `conda run -n kd_mm_beam pytest` 的 paper export、dataset audit、documentation focused tests。
- [x] 4.4 检查 `git status --short --untracked-files=all`，确认未纳入真实数据、论文生成图表、checkpoint、cache、`outputs/` 或 `logs/`。
