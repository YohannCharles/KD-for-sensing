## Context

当前本地实验表明 e5 low encoder LR 是综合候选，e6 hard subset + JEPA 是 robustness-first 候选，BPRR 的无监督 reliability gate 没有打过 e5/e6。下一轮需要在不复制训练框架、不破坏旧输出、不改变默认行为的前提下，一次性扩展已有 PCPG/BPRR 实现，区分 branch weakness 与 routing failure。

## Goals / Non-Goals

**Goals:**
- 用显式 flag 增加 `soft_static` hard subset weighting，并兼容现有 pattern 命名。
- 在已有 fusion/branch 输出上增加 `supervised_router`，用 oracle 或 pattern-best target 监督 router gate，只在 focus patterns 上计算 distill loss。
- 为 GPU1-2 overnight 矩阵提供可恢复 launcher、manifest、失败汇总和每 run 日志。
- 提供 summary 脚本，能合并当前 run 与上一轮 baseline roots，并自动写出 mean/std、delta、e6 来源拆解和 router 诊断。
- 用 focused tests 覆盖新增 helper、dry-run 和 parser。

**Non-Goals:**
- 不新增独立训练框架、长期 package CLI 或旧式兼容入口。
- 不让 oracle label 参与最终预测或真实 ranking。
- 不尝试用 router 解决 radar-only/drop3 单模态 branch weakness。
- 不重写历史 outputs、checkpoint、日志或 baseline 结果。

## Decisions

- **复用现有 local/manual script surface。** Launcher 和 summary 作为 `scripts/` 下的研究脚本保留，不新增 `pyproject.toml` console script；训练 job 仍通过 `conda run -n kd_mm_beam kd-sensing-train` 进入当前训练入口。
- **soft_static 是纯 pattern 权重表。** 权重 helper 对 known aliases 返回固定值，unknown fallback 为 `1.0`，并保证有限数值；训练配置和 summary 记录 `hard_subset_weighting` 字段。
- **supervised_router 是显式 opt-in fusion。** 只有 `--fusion supervised_router` 或等价配置启用时才接入 router distill 与 diagnostics；普通 PCPG、BPRR 和 raw confidence gate 默认不变。
- **oracle target 只来自可用模态。** 训练/验证时根据 unimodal logits 与真实 beam label 计算 beam 距离，tie 时按 CE loss、confidence、固定模态顺序确定，单模态样本 gate 自动为 1 且可跳过 distill。
- **masked softmax 独立成可测试 helper。** 不可用模态 gate 强制为 0，多模态 gate 和为 1，单模态 gate 为 1，避免 NaN。
- **launcher 只调度独立 run。** 每个 job 写单独 log 与 output dir；失败 job 不影响已启动 job，全部结束后再用非零退出报告失败。
- **summary 容忍 artifact 差异。** Parser 优先读取规范 metrics/run config/router diagnostics，也兼容 fake metrics 与历史 baseline roots；缺失字段保持空值，不伪造指标。

## Risks / Trade-offs

- [Risk] 现有训练配置与上一轮脚本的 flag 名称可能不完全一致。→ 先 inspect 当前实现，优先复用已有参数和输出格式，新增 alias 只做显式 opt-in。
- [Risk] oracle target 依赖 unimodal logits，某些 forward 路径可能缺少该字段。→ 缺字段时 fail fast 或降级到 `pattern_best`，并在日志/summary 中显式记录。
- [Risk] overnight job 很长且可能单 job 失败。→ launcher 写 manifest、failed_jobs.csv、每 job log，并支持 `--skip_completed` 和 `--force`。
- [Risk] summary 自动结论可能过度解释。→ 结论只基于聚合指标和阈值化描述，明确区分 routing pattern 改善与 branch weakness。

## Migration Plan

1. 新增 OpenSpec artifact 和 focused tests。
2. 扩展已有 hard subset/router helper 与训练参数解析，默认保持不变。
3. 新增 launcher/summary 脚本，dry-run 验证矩阵和 manifest。
4. 运行 focused pytest、dry-run、smoke test。
5. smoke 通过后用 `--skip_completed` 启动正式 overnight，输出写入 `outputs/overnight_branch_router_v2/`。
