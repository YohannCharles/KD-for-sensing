## Context

仓库已有主线模型目录、协议表、claim registry 和若干 PDF，但论文交付层还缺统一出口：可审阅 claim 到表格/图的导出、数据/复现口径审计、相关工作矩阵。与此同时，`README_REPRODUCE.md` 中仍有旧脚本名漂移风险，需要收敛到当前包内或保留脚本入口。

本设计把论文交付层拆成三块：paper export 只消费已审阅或明确标记状态的结果，dataset audit 只读检查数据/CSV/split，literature matrix 维护文献和本仓库对照关系。

## Goals / Non-Goals

**Goals:**

- 从 claim registry、ledger 或统计/stress summary 生成论文表格/图的 Markdown、CSV 和 LaTeX 草稿。
- 提供 dataset/reproducibility audit，检查 canonical/legacy layout、CSV 字段、模态引用、label range、split leakage 和 official/local blocked 状态。
- 维护 `docs/literature_matrix.md` 与可选 `paper/references.bib`，记录 AMBER、AMR-Net、RMBP-MM、DeepSense6G、BeamBench 等相关工作。
- 更新复现文档，移除或替换已不存在的旧脚本说明。
- 保持生成 artifact 位于 ignored 输出目录。

**Non-Goals:**

- 不生成最终论文 PDF，不强制引入 LaTeX 构建系统。
- 不下载论文、官方数据、官方权重或外部仓库。
- 不自动把 pending/smoke/historical 结果升级为正式表格。
- 不重写真实数据或迁移 `dataset/`。

## Decisions

1. **paper export 只消费 reviewed 或显式状态 rows。**
   默认只允许 `local strict-validation`、`local experimental baseline`、`local substitute` 和明确审阅过的 rows 进入主表。pending/mock/historical 可进入 appendix 草稿，但必须保留状态列。

2. **输出多格式，但核心是机器可读 manifest。**
   每次 export 写 `paper_export_manifest.json`，登记输入 claim、输出表格/图、状态过滤、warnings 和源 artifact。Markdown/CSV/LaTeX 都从同一 normalized table 派生。

3. **dataset audit 不作为训练前置强制项。**
   审计入口帮助排查数据和复现口径，但普通训练入口不强制依赖 audit 成功。

4. **literature matrix 是文档，不是论文管理系统。**
   首版维护简单 Markdown 表和可选 BibTeX，避免引入 Zotero/BetterBibTeX 等外部依赖。

5. **修复文档漂移优先于恢复旧脚本。**
   `README_REPRODUCE.md` 中不存在的 `scripts/check_dataset.py` 等应替换为 current package/module/script 入口或标记历史，不恢复旧入口。

## Risks / Trade-offs

- **Risk:** paper export 消费 pending rows 导致论文表误用。  
  **Mitigation:** 默认拒绝 pending/mock/historical 进入 main tables，除非用户显式 `--include-status`，且输出保留状态列。

- **Risk:** dataset audit 对不同 CSV 变体过严。  
  **Mitigation:** audit 输出 blocked/warning/ok 分级；字段别名通过配置或 dataset descriptor 标准化。

- **Risk:** literature matrix 维护成本上升。  
  **Mitigation:** 只记录与当前主线有关的少量字段，不维护完整综述正文。

- **Risk:** LaTeX 导出格式频繁变。  
  **Mitigation:** 首版只导出简单 tabular/CSV/Markdown，最终排版由论文仓库或人工调整。

## Migration Plan

1. 定义 paper export 输入 schema 和输出 manifest。
2. 实现 claim/ledger/summary 到 table rows 的 normalizer。
3. 实现 dataset audit CLI/模块，覆盖 DeepSense6G/BeamBench/MMW 常见字段和 split leakage 检查。
4. 新增 `docs/literature_matrix.md` 模板和可选 `paper/references.bib`。
5. 更新 README/README_REPRODUCE/主线文档入口，替换不存在脚本说明。
6. 回滚时删除新增 export/audit/literature artifact；训练和评估 workflow 不受影响。

## Open Questions

- paper export 是否放在 `src/kd_sensing/diagnostics/` 还是 `tools/analysis/`；若需要 console script，则放包内 CLI。
- 首版是否需要生成 matplotlib 图，还是只输出图表数据和 manifest。
- literature matrix 的 BibTeX 是否手动维护，还是从 DOI/arXiv ID 可选生成。
