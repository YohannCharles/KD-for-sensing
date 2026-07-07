## Context

P1 候选不是单纯“历史报告”，而是仍可能有诊断价值的包装入口。审计发现：

- `src/kd_sensing/diagnostics/predictive_gps_query_visualizations.py` 与对应 CLI 主要产出解释性 attention/gate/embedding 图，和 predictive robustness diagnostics bundle 高度重叠。
- `src/kd_sensing/cli/plot_mmw_town_gps_v2.py` 与 `compare_mmw_town_gps_v2.py` 是 MMW Town GPS v2 runner 周边薄入口。
- `scripts/profile_training_io.py` 与 `scripts/recommend_parallel_training.py` 属于同一 throughput profiling 决策面。
- `scripts/mmw/prepare_town10_skybridge.py` 与 `scripts/mmw/build_sequence_splits_from_manifest.py` 更像 package preprocess/data owner 的命令配方。

约束：

- 所有 Python 验证命令 MUST 使用 `conda run -n kd_mm_beam <command>`。
- 不改变诊断指标、adapter 训练语义、profiling 字段或数据 split 语义。
- 不新增同职责 wrapper 或旧入口 fallback。

## Goals / Non-Goals

**Goals:**

- 减少 CLI 和 script 数量，同时保留诊断能力。
- 让同一领域的 run、plot、compare、visualize、recommend 模式由一个 owner 管理。
- 更新 help、docs、inventory 和 focused tests，使替代入口清晰。
- 降低未来每个诊断新增一个 CLI 文件的倾向。

**Non-Goals:**

- 不删除仍是 claim evidence generator 的核心诊断逻辑。
- 不重写 MMW Town GPS v2 adapter 或 predictive JEPA benchmark。
- 不把所有诊断塞进一个跨领域 `diagnostics` mega CLI。
- 不处理 P0 本地报告 cluster 或 P2 治理/模型候选。

## Decisions

### Decision 1: 诊断 wrapper 通过 mode 收敛

若一个 CLI 只调用同一 owner 的 plot、compare、visualize 或 recommend 函数，implementation SHOULD 将其合并为 owner CLI 的 subcommand 或 flag。旧 CLI 删除后，help tests 和 docs MUST 指向新命令。

### Decision 2: predictive explanatory visualizations 属于 diagnostics bundle

predictive GPS query visualizations 是解释性补充，不是 claim 的唯一依据。它们 SHOULD 作为 predictive JEPA robustness diagnostics bundle 的可选输出 mode，而不是独立 CLI。

### Decision 3: MMW Town GPS v2 run/plot/compare 同属 adapter owner

MMW Town GPS v2 的 plotter 和 comparator SHOULD 收敛到 adapter v2 owner CLI。若保留单独 Python module 作为内部 helper，外部 current entrypoint 仍 SHOULD 是单一 package CLI。

### Decision 4: throughput profiling 和 recommendation 共享 owner

training IO profile 采样、瓶颈汇总和 parallel recommendation 属于同一决策面。实现 SHOULD 让 recommendation 从 profiling output 或同一 owner 读取字段，而不是保留独立脚本重复解析配置。

## Migration Plan

1. 记录当前 CLI/script help、docs/inventory 引用和 focused tests。
2. 为每组 wrapper 选定 consolidated owner 和 mode 名称。
3. 先扩展 owner CLI 的 help、参数和输出 manifest，再迁移调用点。
4. 删除旧 wrapper/CLI 文件，更新 `pyproject.toml` console scripts、docs 和 tests。
5. 对仍需保留的 helper module 标记为 internal owner helper，不作为 current entrypoint。
6. 运行 focused validation 和 surface guardrail。

Rollback：若某个 wrapper 实际承载独立输入契约、输出 schema 或 claim evidence，保留它并记录 retained-with-reason；不得用旧路径 alias 伪装迁移完成。

## Validation Plan

- `openspec validate p1-consolidate-diagnostic-wrappers --strict`
- `openspec validate --all --strict`
- `conda run -n kd_mm_beam pytest tests/test_cli_help.py -q`
- predictive JEPA diagnostics focused tests。
- MMW Town GPS v2 focused smoke tests。
- training throughput profiling focused tests。
- `PYTHONPATH=src conda run -n kd_mm_beam python -m kd_sensing.cli.project_surface_doctor --scope scripts --scope hotspots --format markdown`

## Open Questions

无阻塞问题。实现阶段需要逐项确认 console script 是否仍有外部文档引用；有引用时先更新文档，再删入口。
