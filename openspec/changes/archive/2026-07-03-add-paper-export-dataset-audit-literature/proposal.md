## Why

研究进入论文写作阶段后，最耗时的往往不是再训练一个模型，而是把已完成实验稳定导出成论文表格/图、确认数据和复现口径没有漂移、并维护相关工作矩阵。这个 change 补论文交付层：paper artifact export、dataset audit 和 literature matrix。

## What Changes

- 新增 paper artifact export 能力，从 claim/ledger/summary 读取已审阅结果，生成论文表格、消融表、stress 曲线、pattern heatmap 和 Markdown/LaTeX/CSV 草稿。
- 新增 dataset/reproducibility audit 入口，替代文档中已漂移的旧 `scripts/check_dataset.py` 等说明，检查数据目录、CSV 字段、模态文件引用、label range、split leakage、scene/sample id 和 local/official reproduction blocked 条件。
- 新增 literature matrix 能力，维护 `docs/literature_matrix.md` 和可选 `paper/references.bib`，记录相关工作的方法、模态、缺失协议、数据集、指标、开源状态、官方复现状态和本仓库对照关系。
- 扩展主线文档，声明 paper export 只能消费已审阅 claim 或明确标记的 pending/mock/historical rows，不自动升级结果。
- 不提交生成的 PDF/PNG/真实表格产物，不下载论文或外部数据，不引入重型排版依赖。

## Capabilities

### New Capabilities

- `paper-artifact-export`: 覆盖论文表格/图/LaTeX/Markdown artifact 的输入、输出、状态标记、产物边界和 claim 审核要求。
- `research-literature-matrix`: 覆盖相关工作矩阵、BibTeX 引用、复现状态、对照关系和文档生命周期。
- `dataset-reproducibility-audit`: 覆盖数据/CSV/模态文件/label/split/official blocked 审计入口和只读边界。

### Modified Capabilities

- `beambench-baseline-reproduction`: 用当前包内/脚本入口替代已经漂移的旧脚本说明，并强化 official blocked / local substitute audit 输出。
- `dataset-directory-layout`: 增加 dataset audit 对 canonical/legacy layout 的检查要求。
- `mainline-experiment-documentation`: 增加 paper export 与 literature matrix 的索引、claim 状态和文档生命周期要求。

## Impact

- 主要影响 `src/kd_sensing/diagnostics/` 或 `src/kd_sensing/cli/` 的 paper/audit 入口、`docs/` 文献矩阵、README/复现文档和 focused tests。
- 生成产物默认写入 ignored `outputs/paper_artifacts/`、`outputs/analysis/dataset_audit/` 或显式本地路径；源码只保留模板、实现、测试和摘要文档。
- 所有项目相关 Python 验证命令使用 `conda run -n kd_mm_beam ...`。
