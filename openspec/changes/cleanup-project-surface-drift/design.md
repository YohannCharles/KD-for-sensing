## Context

本 change 面向一次支持面收敛，而不是新增训练能力。当前可观测问题包括：`tests/test_architecture_boundaries.py` 中配置数量和 OpenSpec Purpose hygiene 红点、`docs/project_surface_inventory.md` 中 `configs/fusion/` 数量与真实仓库不一致、`scripts/run_csi_hardening_matrix.sh` 引用已不存在的 hardening matrix 配置名，以及本地 ignored 产物中存在大量 bytecode、egg-info、历史 zip 和输出目录。

项目已有 `project-surface-cleanup` 与 `project-architecture` 约束，且已经要求本地实验产物通过 manifest 或用户显式确认删除。因此本 change 应以“修复漂移并明确可删边界”为主，避免在同一批次引入模型、数据或训练契约变化。

## Goals / Non-Goals

**Goals:**

- 让架构边界测试重新反映当前真实支持面，并消除已知失败项。
- 将 `configs/fusion/` 根目录收敛为长期可维护的 canonical 配置集合，实验特化配置必须有明确归属。
- 修复已发现的脚本、文档和 OpenSpec 当前 specs 中的过期引用或脚手架占位。
- 对无调用源码候选做保守处置：只有确认不属于公共 API、无当前入口依赖且测试覆盖充分时才删除。
- 清理 ignored 的本地临时产物时保持源码与运行产物边界清晰。

**Non-Goals:**

- 不新增训练任务、模型结构、数据集 schema、评估指标或外部依赖。
- 不自动删除 `outputs/`、`logs/`、cache、checkpoint 或真实数据目录中的实验产物。
- 不把历史 archive 重写成当前支持文档。
- 不在本 change 中大规模拆分热点长文件；只处理与支持面漂移直接相关的冗余。

## Decisions

1. **先修确定性红点，再做收缩。**
   - 方案：先修 `gps-conditioned-jepa-pretraining` 的 Purpose、hardening matrix 脚本引用和 inventory 统计，再处理配置收缩。
   - 理由：这些问题有明确验收信号，能快速恢复 guardrail 的可信度。
   - 备选：先删除配置和源码候选。该方案风险较高，因为现有文档/测试仍漂移，容易误判当前支持入口。

2. **`configs/fusion/` 根目录只保留 canonical 或文档声明的当前入口。**
   - 方案：逐个审计 23 个 YAML，将长期入口保留在根目录；实验矩阵、低内存复现实验、best/last 对照等特化配置迁移到明确实验子目录、归档或删除，并同步引用。
   - 理由：根目录过宽会让用户无法判断推荐入口，也让架构测试失去约束力。
   - 备选：提高测试阈值到 23。该方案只能掩盖漂移，不减少维护面。

3. **源码删除采用“公共 API 反证”策略。**
   - 方案：对 CodeGraph 显示无调用的候选，再检查 `pyproject.toml`、README、docs、OpenSpec、测试和 `__init__.py` 是否声明为入口；只有全部不命中时才删除或降级为内部实现。
   - 理由：无 AST 调用不等于无外部用户，尤其是 CLI、动态注册和公开 helper。
   - 备选：按无调用结果直接删除。该方案容易破坏外部脚本或研究复现实验。

4. **本地产物清理与源码变更分层处理。**
   - 方案：`__pycache__`、`.pytest_cache`、egg-info 等可作为本地临时产物清理；`outputs/`、`logs/`、cache 和 checkpoint 继续依赖 cleanup manifest 或用户显式确认。
   - 理由：源码清理和实验结果清理的风险模型不同，应保持可审计边界。
   - 备选：在同一任务中删除所有 ignored 目录。该方案可能清掉仍有分析价值的实验结果。

## Risks / Trade-offs

- **[Risk] 配置迁移后文档或脚本仍引用旧路径。** → 使用 `rg` 扫描被迁移文件名，并运行架构边界测试与相关 CLI help。
- **[Risk] 删除无调用 helper 破坏外部用户脚本。** → 删除前检查公开导出、console scripts、文档和 OpenSpec 声明；不确定时先保留并记录后续收敛项。
- **[Risk] 只修测试阈值而没有减少维护面。** → 验收要求必须同时更新 inventory，并说明保留/迁移/删除的配置分类。
- **[Risk] 本地产物清理误伤实验结果。** → 只默认清理 bytecode/cache 元数据；真实实验输出必须走 manifest 或用户确认。

## Migration Plan

1. 修复 OpenSpec Purpose、脚本过期路径和 inventory 统计漂移。
2. 审计并收缩 `configs/fusion/`，同步所有源码、文档和测试引用。
3. 审计无调用源码候选，按公共 API 反证策略删除或保留说明。
4. 清理低风险 ignored 本地产物，并报告未自动删除的实验输出候选。
5. 运行 `openspec validate cleanup-project-surface-drift --strict`、`openspec validate --all --strict` 和 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。

回滚策略：源码与文档修改可按 Git diff 回滚；配置删除或迁移必须在同一 change 中保留可追踪 diff；本地 ignored 产物清理不纳入源码回滚，且不应包含真实实验输出的自动删除。

## Open Questions

- `configs/fusion/` 中哪些 JEPA lowmem/best/last 配置仍需作为长期入口保留，需要在实现审计时结合 README、实验矩阵和近期脚本引用确认。
- CodeGraph 显示无调用的 `evaluation` 与 LOSO helper 是否被用户外部脚本依赖，若无法确认，应优先保留并只更新 inventory。
