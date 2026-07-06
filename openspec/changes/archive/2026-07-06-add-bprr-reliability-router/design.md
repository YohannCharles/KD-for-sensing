## Context

当前缺失模态实验已经有 `u_mask_beam_jepa`、weighted_sum / reliability / PCPG 融合、unimodal diagnostics、branch aux、hard subset weighting、missing-aware checkpoint selection、Scene31-34 fresh eval 输出和 `outputs/pcpg_radar_balance_v1` 本地实验脚本。上一轮结果说明低 encoder LR 和 hard subset / JEPA 对不同 missing pattern 各有收益，但仍缺 oracle 上界，以及可校准、pattern-aware、prototype-aware 的全局 router。

本 change 不替换当前主线，也不把 reliability fusion 晋升为 final method；它只新增本地可靠性路由实验面，用于初筛 BPRR 是否值得多 seed。所有新行为必须显式 opt-in，默认训练、评估、checkpoint 和旧脚本保持不变。

## Goals / Non-Goals

**Goals:**

- 在 U-MaskBeamJEPA 缺失模态路径中新增 `fusion_type: raw_conf_gate` 和 `fusion_type: bprr`，优先支持 logits 层融合。
- 复用上一轮 PCPG 的 unimodal logits/prototype diagnostics、hard subset weighting、JEPA alignment 和 checkpoint selection，不复制训练框架。
- 提供 BPRR reliability feature 构造、per-modality temperature calibration、gate balance regularization、radar gate floor regularization 和 gate diagnostics。
- 补齐 eval-only oracle gate，并提供 `e3/e7/e8/e9/e10/e11/e12` launcher 与 summary。
- 用 focused tests 固定 mask、calibration、regularization、oracle、launcher 和 summary 行为。

**Non-Goals:**

- 不新增完整模型注册名、package console script、长期 public API 或旧式 root training script。
- 不默认启用 raw confidence gate、BPRR、calibration、balance reg、radar floor reg、hard subset 或 JEPA alignment。
- 第一版不强制接入真实 prototype distance；如果当前模型没有稳定 distance 输出，可在统一 feature 接口中保留 `None` / zero fallback 和 TODO。
- 不提交真实 `outputs/`、日志、checkpoint、generated config 或 dataset 内容。

## Decisions

1. **BPRR 放在 U-MaskBeamJEPA / JEPA fusion owner 内部，优先 logits 融合。**  
   现有模型已持有 per-modality branch、available mask、beam head、prototype bank 和 PCPG diagnostics，是最小侵入点。BPRR 输出 `[B, M]` gate 后对 unimodal logits 做加权和。备选方案是在通用 trainer 中实现 fusion，但会把模型私有 diagnostics 泄漏到训练循环。

2. **raw confidence gate 与 BPRR 共用 mask / diagnostics helper。**  
   raw gate 使用 margin / temperature 的简单 masked softmax，作为不校准、不平衡保护的 baseline。BPRR 使用全局 reliability feature 向量、pattern bias 和 router MLP，便于比较普通 confidence gate 与 pattern-aware router。备选方案是只做 MLP router，但会失去必要反例。

3. **calibration 第一版使用 per-modality temperature。**  
   temperature 通过 softplus 保证为正，默认初始化为 `bprr_init_temperature`。为避免改变最终分类器校准口径，第一版优先将 temperature 作用在 router reliability feature 的 logits-derived 统计；如果实现中也用于最终 logits，必须在代码注释和 diagnostics 中写清。备选 learned_affine 保留参数入口，可先报清晰不支持或作为 temperature 的扩展。

4. **gate regularization 作为训练 extension / model extra loss 输出。**  
   balance reg 和 radar floor reg 只在训练启用，eval 不强制 gate floor。实现应复用已有 extension pattern 或模型 diagnostics 中的 `extra_losses`，避免直接改普通 base loss。备选直接写入 trainer 主 loss 逻辑，改动面更大。

5. **launcher 和 summary 保持 local/manual script。**  
   新脚本只负责本地实验编排和只读聚合，不进入 `pyproject.toml`。launcher dry-run 也写 manifest，正式运行每个 job 独立 log，失败 job 不杀掉已启动 job。summary 兼容 fake metrics 以便测试，并合并 `outputs/pcpg_radar_balance_v1` 中 e5/e6 baseline。

## Risks / Trade-offs

- **prototype distance 可能拿不到或语义不稳定。** → 第一版以 logits-derived reliability 为可运行 fallback，接口保留 `prototype_min_distance` / `prototype_margin` 字段和 TODO。
- **BPRR 仍可能学成 image/lidar gate 塌缩。** → 输出 pattern-level gate diagnostics，并提供 balance reg 与 radar floor reg 两个显式 ablation。
- **oracle gate 容易被误作真实方法。** → 输出必须标注 `oracle`，summary 不把 oracle 混入真实方法排名。
- **训练矩阵耗时且依赖本地 checkpoint。** → e3 找不到 e5 checkpoint 时 fail fast；launcher 支持 dry-run、skip_completed、force 和 max_epochs smoke。
- **新增 local/manual scripts 可能触发表面积检查。** → scripts 保持研究脚本属性，测试覆盖 dry-run/summary，不新增 package CLI。

## Migration Plan

1. 创建 opt-in BPRR/raw gate/oracle/config 参数和 focused tests，默认行为不变。
2. 新增 launcher、summary 和 fake artifact tests。
3. 运行 `openspec validate add-bprr-reliability-router --strict`、focused pytest、dry-run 和 smoke。
4. 回滚时删除新增 opt-in helper、scripts/tests 和 OpenSpec change；旧 PCPG、训练默认和既有 outputs 不需要迁移。

## Open Questions

- 当前 prototype bank 是否能稳定输出 per-modality distance/margin；若不能，本轮仅记录 logits fallback。
- `learned_affine` calibration 是否需要在第一轮就支持；若实现风险超过收益，可先保留参数但报清晰错误。
