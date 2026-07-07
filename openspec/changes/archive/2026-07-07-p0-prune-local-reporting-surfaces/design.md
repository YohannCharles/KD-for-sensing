## Context

P0 对应 ponytail-audit 中“最大、最像一次性研究脚手架”的部分。审计候选包括：

- `scripts/analysis/*.py` 五个研究分析脚本，约 2285 行。
- Scene31-34 final analysis cluster，约 4973 行。
- Scene31 单场景 summary cluster，约 3466 行。
- `scripts/reevaluate_apples_to_apples.py`，约 988 行。

这些文件多数围绕结果汇总、论文表格、展示材料、补充诊断或某一轮实验复评。它们的价值在于已经生成的结论和可复现命令，而不是永久保留每个临时 Python 入口。

约束：

- 所有 Python 验证命令 MUST 使用 `conda run -n kd_mm_beam <command>`。
- 不删除本地数据、运行输出、日志、cache、checkpoint 或历史权重。
- 不新增 alias、compat wrapper、deprecation trampoline 或旧脚本 fallback。
- 删除脚本前 MUST 先确认 current docs/spec/tests 不再要求该路径，并保留必要结论证据。

## Goals / Non-Goals

**Goals:**

- 最大幅度减少本地报告和一次性分析脚本数量。
- 将仍有价值的输出契约迁到少数 owner 命令或 package-local helpers。
- 保留论文表格、claim notes、关键 CSV/Markdown/figure 输出的字段语义和可验证性。
- 更新 inventory 和 guardrail，防止旧脚本或等价 wrapper 回流。

**Non-Goals:**

- 不改变训练、评估、模型、数据 split 或指标定义。
- 不重跑完整实验或重新生成正式结果。
- 不把多个脚本简单搬进一个更大的 `utils` 或 `analysis_common` 杂物间。
- 不处理 P1 诊断 wrapper 和 P2 治理/模型候选；它们分别由独立 change 管理。

## Decisions

### Decision 1: 一次性研究脚本默认退出 current surface

`scripts/analysis/` 下的脚本若只用于生成已沉淀的支持材料，implementation SHOULD 删除源码入口，并在 docs 或 retained artifact note 中保留：

- 对应结论或表格位置；
- 原始输入 artifact 的期望路径模式；
- 若需要复核，推荐的 canonical command 或手动复现说明。

理由：这类脚本最容易过期，且通常没有通用 CLI 语义。把它们留在 current surface 会让维护者为历史探索脚本支付持续成本。

### Decision 2: Scene31-34 final analysis 只能有一个 owner

Scene31-34 final analysis 当前拆成多个 profile、significance、paper table、conclusion、CDF、heatmap、sampling 和 presentation exporter。实现 SHOULD 收敛为一个 owner 命令或一个 owner module 下的窄子命令，所有输出通过显式 `--artifact`、`--view` 或 profile 参数表达。

替代方案是保留多个脚本但共享 common helper。这个方案保留了大部分入口面，不能解决 current surface 膨胀。

### Decision 3: Scene31 单场景 summary 使用参数化 owner

Scene31 的 BC-next、P0 fresh eval、baseline pack、subset reliability、patternfilm、funnel、next-round 和 subset reference summary SHOULD 共享一个 summary owner。profile/group 决定输入 glob、输出 schema 和标签，而不是通过文件名复制控制流。

实现时 MUST 保持每个 workflow 已承诺的输出字段、排序、异常信息和默认路径，除非对应 current spec 同步修改。

### Decision 4: apples-to-apples 复评回到 evaluate workflow

`scripts/reevaluate_apples_to_apples.py` 若仍是 current need，SHOULD 折入 package evaluate workflow、已有 CLI 或 narrow eval helper。若只是历史复评脚本，SHOULD 删除并在 docs 中保留替代命令。

理由：评估行为属于 package evaluation contract，不应该由根目录大型一次性脚本长期拥有。

## Migration Plan

1. 记录 baseline：`git status --short`、脚本列表、当前 docs/spec/tests 引用。
2. 对每个候选脚本标注：delete、consolidate、retained-with-reason。
3. 先更新 OpenSpec current spec、README/docs 和 inventory，使 current contract 指向 owner 命令。
4. 删除 `scripts/analysis/` 中已沉淀结论的一次性脚本。
5. 合并 Scene31-34 final analysis 输出 owner，删除 per-artifact 脚本。
6. 合并 Scene31 单场景 summary owner，删除重复 summary 脚本。
7. 将 apples-to-apples fresh eval 行为折回 evaluate workflow 或删除历史脚本。
8. 运行 focused validation 和 surface guardrail。

Rollback：若删除后发现仍有 current spec 或 focused test 依赖旧路径，恢复该文件并记录 retained-with-reason；不得通过新增 wrapper 掩盖 contract 漂移。

## Validation Plan

- `openspec validate p0-prune-local-reporting-surfaces --strict`
- `openspec validate --all --strict`
- `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
- `conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_config_load_characterization.py -q`
- 与 Scene31/Scene31-34 summary、paper table、claim export 相关的 focused tests。
- `PYTHONPATH=src conda run -n kd_mm_beam python -m kd_sensing.cli.project_surface_doctor --scope scripts --format markdown`

## Open Questions

无阻塞问题。实现阶段唯一需要逐项确认的是：某个脚本是否仍是 current evidence generator。若答案不清楚，默认先在 inventory 中标记 retained-with-reason，而不是盲删。
