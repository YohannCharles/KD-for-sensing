## 1. 清理 manifest 基础能力

- [x] 1.1 新增运行产物清理数据模型，覆盖 manifest metadata、候选项、保护项、规则 ID、风险等级、大小和 mtime 字段。
- [x] 1.2 实现只读候选扫描 helper，支持扫描 `outputs/`、`logs/`、cache、`.pytest_cache/` 和 `__pycache__/`，默认不删除、不移动、不压缩、不重写文件。
- [x] 1.3 实现保护规则，默认保护 git 已跟踪文件、`dataset/`、`All_models/`、`src/`、`configs/`、`docs/`、`openspec/`、`tests/` 和活跃运行。
- [x] 1.4 实现候选分类规则，覆盖 Python bytecode、pytest cache、`_debug`、`_plan_check`、`outputs/other/`、失败/stale run、重复 checkpoint、日志目录和个人备份压缩包。
- [x] 1.5 增加 JSON manifest 写出和摘要统计，记录扫描根、规则版本、候选数量、候选总大小、保护数量和生成时间。

## 2. Run Index 与 Checkpoint Retention

- [x] 2.1 扩展 `kd_sensing.diagnostics.run_index` 的 run summary，加入 run 目录大小、checkpoint 数量、checkpoint 总大小和主要 checkpoint 路径。
- [x] 2.2 增加 checkpoint retention 摘要，记录 checkpoint 来源、selection metadata、sidecar 是否存在、normalization artifact 引用和 registry 默认候选状态。
- [x] 2.3 将 `running`、`waiting`、最近更新且未完成的 run 暴露为清理保护信号。
- [x] 2.4 将 stale、failed、partial run 的缺失 artifact、日志关联和候选理由暴露给清理 manifest。
- [x] 2.5 为 `last.pth`、重复 probe checkpoint 和失败 run 临时 checkpoint 生成保守候选，不默认候选 `best.pth`、`best_top1.pth` 或 sidecar metadata。

## 3. CLI 与删除阶段

- [x] 3.1 新增或扩展包内 CLI，提供 dry-run manifest 生成入口，默认输出到 `outputs/cleanup_manifests/` 或用户显式路径。
- [x] 3.2 在 `pyproject.toml` 声明对应 console script，并确保 editable install 后入口可用。
- [x] 3.3 实现基于 manifest 的显式 apply/delete 入口，要求传入 manifest 和确认参数；未确认时必须拒绝删除。
- [x] 3.4 删除阶段在执行前重新验证路径仍位于允许根、未被 git 跟踪、未受保护且状态与 manifest 兼容。
- [x] 3.5 删除阶段写出结果报告，记录已删除、跳过、失败和保护状态变化的路径。

## 4. 输出目录与脚本表面积收口

- [x] 4.1 将 `scripts/run_mmw_sunny_modal15_l5p6_h246.sh` 默认 `OUTPUT_ROOT` 从 `outputs/other` 改为语义化 MMW modal15 输出目录。
- [x] 4.2 同步更新 `scripts/run_mmw_sunny_modal15_l5p3_h123.sh` 和 l5p6 脚本帮助文本中的输出目录说明。
- [x] 4.3 更新 `docs/project_surface_inventory.md`，记录清理 manifest 工作流、输出目录命名约定和 `outputs/other/` 历史候选定位。
- [x] 4.4 更新 README 或扩展指南中的本地产物边界说明，明确清理必须先生成 manifest 且删除需要显式确认。
- [x] 4.5 更新架构边界测试，拒绝长期保留脚本新增 `outputs/other` 默认输出根，并继续拒绝跟踪本地产物。

## 5. 测试与验证

- [x] 5.1 添加清理 manifest 单元测试，覆盖 protected 路径、tracked 文件、Python cache、`outputs/other/`、`_debug`/`_plan_check` 和 checkpoint retention 候选。
- [x] 5.2 添加 run index 扩展测试，验证大小统计、checkpoint 摘要和活跃运行保护信号。
- [x] 5.3 添加 CLI help 测试，使用 `conda run -n kd_mm_beam <cleanup-console-script> --help` 验证入口可用。
- [x] 5.4 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [x] 5.5 运行与新增测试相关的 pytest 子集，例如 `conda run -n kd_mm_beam pytest tests/test_run_index.py tests/test_architecture_boundaries.py -q`。
- [x] 5.6 运行 `openspec validate clean-runtime-artifacts-and-output-surface --strict`。
