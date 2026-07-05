## Context

`scene31_baseline_pack_lmdb` 已完成 all_core，但 AMR-lite / AMBER-lite 的 full 与多个 missing pattern 指标逐项完全相同。当前链路中 fresh eval 已构造 `missing_mask`，但 batch runtime 会按 `model.forward` 签名过滤 kwargs，而 `ModularSequenceModel.forward` 未声明 missing-mask 相关参数，导致 modular lite baseline 实际在 full 输入上重复评估。

同时，可信 proto 结果显示 `proto_randomdrop_subset_es40` 是当前最强缺失模态鲁棒 proto baseline，旧 `proto_sampler_uniform_es40` 只能作为 ablation。后续 reliability fusion 和 PatternFiLM d8 应围绕 randomdrop subset reference 继续，而不是继续扩大 uniform 分支。

## Goals / Non-Goals

**Goals:**

- 修复 modular model missing mask 传递与应用，让 AMR-lite / AMBER-lite fresh eval 的 pattern mask 真实影响 logits。
- 提供诊断脚本和 mask_suspect 标记，避免指标完全相同的结果静默进入正式 ranking。
- 将 summary、delta、promotion 和 runner 默认 reference 切换到 `proto_randomdrop_subset_es40`。
- 审计并复用已有 reliability/mask weighted fusion；缺失时只补最小 proto-compatible 实现。
- 新增 subset reliability 与 subset PatternFiLM d8 的 local/manual 配置、runner 和 combined summary。

**Non-Goals:**

- 不重训 AMR-lite / AMBER-lite 旧 run；只加载已有 best checkpoint 重跑 fresh eval。
- 不删除、覆盖旧 checkpoint 或旧结果 root。
- 不引入 transformer/imputation 新方向，不继续 JTT/MVFR/MPDRO/beamsoft/condBTAPA/weakKD。
- 不新增 package CLI；所有新增执行面保持 `scripts/` local/manual workflow。

## Decisions

1. **把 missing-mask 参数纳入 `ModularSequenceModel.forward`，并在 fusion/core 前应用。**  
   这样 `engine.batch` 的签名过滤会保留 fresh eval 传入的 mask。应用点放在 projected modality feature 收集后、representation core 之前：缺失模态 feature 置零或从 stack 中通过 availability mask 屏蔽，并保留 `missing_modality_metadata` 到输出 diagnostics。替代方案是在 eval 脚本原地改 batch tensor，但那会污染 batch 语义并复制 mask 逻辑。

2. **诊断使用真实 checkpoint 优先，无法加载时保留静态签名诊断。**  
   `scripts/diagnose_modular_missing_mask.py` 读取 run config、best checkpoint 和少量 batch 执行 full vs missing pattern logits 对比；如果本地缺少数据或 checkpoint，脚本输出明确 `diagnosis` 与 warning，而不是伪造 ok。

3. **fresh eval maskfix runner 只做重评，不做训练。**  
   `eval_modular_lite_maskfix` group 只选择已 complete 且存在 best checkpoint 的 AMR/AMBER-lite run，调用现有 apples-to-apples fresh eval helper，不传 `--max-batches`。评估后用 pattern-wise metrics 检查是否 full 与 missing 完全相同，写出 `mask_suspect=true`。

4. **subset reference 是 summary 的唯一默认主 reference。**  
   summary 优先读取 `proto_randomdrop_subset_es40` 的实际 n>=3 fresh eval；不足时使用固定 fallback 并 warning。`proto_sampler_uniform_es40` 保留为 ablation 行，不再作为 delta 或 winner 默认 reference。

5. **reliability fusion 先审计再最小补齐。**  
   若已有 reliability/mask weighted fusion 可兼容 proto 与 randomdrop subset，复用现有模块和配置字段；否则在 modular/proto 路径新增轻量 weight head：对每个模态 pooled feature 加 availability 产生 score，对缺失模态 hard mask 为 0，对可用模态 softmax 归一化。日志只按 epoch 聚合写 `reliability_weights_epoch.csv`。

6. **new candidate runner 仍是 Scene31 local/manual surface。**  
   `scripts/run_scene31_subset_reliability.sh` 按 GPU worker 串行调度 train/eval，训练仍通过 `conda run -n kd_mm_beam kd-sensing-train --config <yaml>`，fresh eval 复用现有 helper。runner 只负责 group 展开、skip/overwrite、日志和失败列表。

## Risks / Trade-offs

- **旧 AMR/AMBER-lite checkpoint 若训练期未学习 mask-aware 行为，修复后结果可能下降** → 这是正确暴露真实缺模态性能；summary 会保留 local experimental baseline caveat。
- **本地没有完整 dataset/checkpoint 时无法在源码验证中跑真实 fresh eval** → 诊断和 runner 支持明确 warning；focused tests 用 synthetic batch 覆盖 forward mask 生效、summary mask_suspect 过滤和 config sanity。
- **reliability fusion 容易过度实现** → 只允许小参数量 mask-weighted fusion，不新增复杂 transformer、imputation 或外部依赖。
- **summary 读取历史产物格式不一致** → reader 宽容解析已知 CSV/JSON，缺关键字段时标 warning 并排除正式 ranking。

## Migration Plan

1. 新增 OpenSpec delta、forward mask 修复、诊断脚本、runner、summary 和 focused tests。
2. 运行 `openspec validate fix-scene31-subset-reference-mask-eval --strict` 与相关 `conda run -n kd_mm_beam pytest ...`。
3. 本地需要真实结果时，先运行诊断，再执行 maskfix fresh eval、reliability/subset_film 训练与 combined summary。
4. 回滚时删除新增脚本/配置和 forward mask 应用改动；旧 outputs、checkpoint 和 baseline pack 结果不受影响。

## Open Questions

- 真实 AMR/AMBER-lite maskfix fresh eval、reliability fusion 三 seed、subset PatternFiLM 三 seed 是否能在当前会话内完成，取决于本地 GPU 和数据可用性；实现必须给出可续跑命令和明确未运行状态。
