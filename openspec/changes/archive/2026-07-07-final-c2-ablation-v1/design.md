## Context

当前代码已经包含 U-MaskBeamJEPA、PCPG、BPRR、supervised router、soft_static hard subset、TinyViT base config、missing-pattern eval 和上一轮 overnight launcher/summary。最终消融应复用这些 owner：模型行为留在 `u_mask_beam_jepa`，训练额外 loss 留在现有 extension，实验编排只做薄 Python launcher，summary 只读 ignored `outputs/`。

## Goals / Non-Goals

**Goals:**
- 用显式 flag 表达 c2 及其消融，默认训练和旧脚本行为不变。
- 补齐 `average` fusion、router feature ablation、prototype/head ablation 和 summary 诊断字段。
- 生成 67 个 final jobs 的可审计 manifest，并能 dry-run/smoke/skip/force。
- 汇总新旧 roots，输出论文所需主表、消融表和自动结论。

**Non-Goals:**
- 不新增完整训练框架、package console script 或长期公共 API。
- 不复写旧 outputs，不把 checkpoint/log/cache 纳入源码。
- 不继续扩展新方法，只验证已冻结的 c2 模块取舍。

## Decisions

- **复用 YAML 生成而不是新增配置族实体文件。** Launcher 从上一轮 TinyViT base config 读入并合并 override，避免提交 67 个可再生成 YAML。
- **模型开关放在 `model.primary`，训练 loss 开关同步写入 `loss`/`training`。** 这样 direct CLI override、generated config 和最终 `final_config.yaml` 都能审计。
- **`head_type=classifier` 保留 prototype bank 模块但不使用 prototype logits/损失/特征。** 这是最小兼容实现；router prototype margin 自动置零并在 diagnostics 记录 fallback。
- **`average` fusion 只做 available logits mean。** 单模态可用时自然等于该模态，所有不可用模态被 mask，不引入额外参数。
- **Summary 使用宽容 parser。** 兼容 `eval/*_missing_patterns.csv`、`eval_matrix.csv`、run config sidecar 和历史 baseline roots；缺字段保留空值，不阻断其它表输出。

## Risks / Trade-offs

- [训练耗时长] → launcher 支持 dry-run、max_epochs smoke、skip_completed 和 failed_jobs，不要求单次交互等待 67 个长训完成。
- [base config seed 文件缺失] → 沿用上一轮 seed1 fallback，dry-run 允许占位，正式运行 fail fast。
- [prototype margin 依赖 head_type] → classifier 或禁用 prototype 时显式 fallback 到零 margin，并写入 diagnostics/metadata，避免 silent failure。
- [summary 输入形态漂移] → parser 优先按已有 overnight summary 的模式聚合，测试用 fake metrics 覆盖核心路径。
