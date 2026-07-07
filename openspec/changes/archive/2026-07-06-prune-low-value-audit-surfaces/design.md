## Context

本 change 是 ponytail-audit 后的低价值表面收缩。仓库已经有 `right-size-public-entrypoint-surface` 和 `shrink-experiment-config-families` 两个完成但未归档的 change；它们是 closeout 候选，不应把归档动作混入本 change。这里处理的是剩余代码和脚本层面的“小包装、小门面、小样板”。

约束：

- 所有 Python 验证命令 MUST 使用 `conda run -n kd_mm_beam <command>`。
- 不新增 alias、compat wrapper、deprecation trampoline 或旧入口 fallback。
- 不触碰 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、`All_models/` 或本地验证产物。
- 不删除已明确保留的当前 owner：`canonical_virtual.py`、runtime artifact cleanup、JEPA GPS shortcut benchmark owner、TinyViT 4 registry names 和 canonical YAML 配置族。

## Goals / Non-Goals

**Goals:**

- 删除或收缩审计确认的 package facade、薄 script wrapper、小型 export 聚合模块和测试 path 样板。
- 让内部源码导入真实 owner module，而不是通过 package barrel 或 facade 省 import 行。
- 更新 inventory、architecture guardrail 和 focused tests，防止同名包装回流。
- 对可能没有净收益的 descriptor dataclass 简化设置明确闸门：只有代码更少、语义更清楚、测试不变时才改。

**Non-Goals:**

- 不改变训练、评估、数据 split、指标或 JEPA/GPS benchmark 语义。
- 不做全仓 import 风格美化，不批量改无关 `__all__`。
- 不合并 active complete change，也不替它们执行 archive。
- 不生成配置、不重排 267 个 YAML，不新增统一 utils 包。

## Decisions

### Decision 1: 一个 change 覆盖本轮审计删减

本轮候选共享同一主题：删除低价值包装层，并且都需要同步 project surface inventory、architecture boundary 和 focused tests。使用一个 change 可以避免四五个小 change 互相抢同一份 docs/inventory。

替代方案：为 difficulty facade、Scene31 scripts、eval export 和测试样板分别建 change。这个方案管理成本高，且每个 change 的验证集合高度重复。

### Decision 2: owner import 替代 package facade

`kd_sensing.data.difficulty.__init__` 只 re-export schema 和 pipeline 符号，并通过 lazy `__getattr__` 继续维持 barrel 行为。内部源码和测试 MUST 改为：

- schema 类型/函数从 `kd_sensing.data.difficulty.schema` 导入；
- pipeline 构建/解析函数从 `kd_sensing.data.difficulty.pipeline` 导入；
- 默认 operator 注册继续由 `kd_sensing.data.difficulty.operators` 包负责。

理由：difficulty 不是外部稳定 public import 契约；内部代码通过 facade 会隐藏真实 owner 并扩大子包 marker 维护面。

### Decision 3: Scene31 wrapper 只保留 canonical command

以下 wrapper 只设置默认参数或转发到已有 canonical 命令，应删除：

- `scripts/summarize_scene31_beamsoft_weak.py` → `scripts/summarize_scene31_bc_next.py --root ...`
- `scripts/run_scene31_modular_maskfix_eval.sh` → `scripts/run_scene31_subset_reliability.sh --group eval_modular_lite_maskfix`
- `scripts/run_scene31_baseline_pack_maskfix_eval.sh` → `scripts/run_scene31_subset_reliability.sh --group eval_modular_lite_maskfix`

docs/tests/inventory MUST 指向 canonical command。历史说明 MAY 保留，但必须标明旧 path 已删除或 retired。

### Decision 4: eval writer 回到 eval matrix owner

`kd_sensing.eval.export` 只承载 U-mask Beam JEPA eval matrix 的 CSV/JSON/Markdown writer，被 CLI、trainer runtime helper 和相关测试使用。实现 SHOULD 将 writer 合并到 `kd_sensing.eval.u_mask_beam_jepa_eval_matrix` 或该 owner 附近的窄私有 helper，调用方从 eval matrix owner 导入，随后删除 `export.py`。

理由：CSV/JSON/Markdown writer 不是跨领域 eval API。保留 `kd_sensing.eval.export` 会像通用杂物间一样诱导后续 helper 继续堆进去。

### Decision 5: 微缩只做确定收益

两类低风险微缩纳入本 change：

- `gps_query_evidence` 的 `json.loads(json.dumps(...))` 改为 `copy.deepcopy(...)`。
- 删除 tests 中由 `tests/conftest.py` 已统一处理的 `ROOT/SRC` sys.path 样板和对应 `E402` 噪声。

`dataset_descriptors` 只做 gated simplification：如果静态 mapping + query function 能保持现有 API、错误信息和测试语义，并明显减少 dataclass/转换样板，则实施；否则只更新 inventory 保留理由和未来删除触发条件。

## Migration Plan

1. 记录 baseline：`git status --short`，确认本 change 不包含本地产物。
2. 迁移 difficulty import 调用点到 `schema` / `pipeline` owner；收缩 `data/difficulty/__init__.py`；补 architecture check。
3. 删除 Scene31 wrapper，更新 docs/inventory/tests 中的旧 path；确认 canonical command 文档足够。
4. 合并 eval writer 到 U-mask Beam JEPA eval matrix owner，删除 `src/kd_sensing/eval/export.py`，更新 CLI/trainer/tests import。
5. 执行微缩：`copy.deepcopy` 替换 JSON deep copy；删除重复 test path boilerplate。
6. 尝试 `dataset_descriptors` gated simplification；若没有净收益则不改实现，只补保留理由。
7. 运行 focused validation，再运行 OpenSpec validation 和 project surface doctor。

Rollback：若某个删除项被发现仍有 current docs/spec/tests 契约消费，恢复该文件并在 inventory 中登记保留理由；不得用新 wrapper 或 alias 代替原路径。

## Validation Plan

- `openspec validate prune-low-value-audit-surfaces --strict`
- `openspec validate --all --strict`
- `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
- `conda run -n kd_mm_beam pytest tests/test_modality_difficulty.py -q`
- `conda run -n kd_mm_beam pytest tests/test_u_mask_beam_jepa_eval_matrix.py -q`
- Scene31 wrapper 删除后运行相关 focused tests，例如 `tests/test_scene31_next_round.py`、`tests/test_scene31_baseline_pack.py` 中仍存在的无副作用检查。
- 若触碰 dataset descriptors：`conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py tests/test_training_io_dataset_workflow.py -q`
- Surface guardrail：`PYTHONPATH=src conda run -n kd_mm_beam python -m kd_sensing.cli.project_surface_doctor --scope scripts --scope hotspots --scope closeout --format markdown`

## Open Questions

无阻塞问题。实现阶段唯一需要按证据决定的是 `dataset_descriptors` 是否真的变短变清楚；若不能同时满足“更少代码”和“同等错误语义”，默认保留现状。
