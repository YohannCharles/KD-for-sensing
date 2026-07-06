## Context

当前缺失模态主线已经有 `u_mask_beam_jepa`、weighted_sum / reliability 融合、missing-mask forward、训练 extension、Scene31/31-34 本地 runner 和 summary surface。用户本轮目标不是替换主线，而是在 TinyViT 强 image/lidar 后验证 radar 分支被共享表示/共享头边缘化的假设，因此所有新机制必须通过显式配置开启，并复用现有 `kd-sensing-train`、`kd-sensing-evaluate`、checkpoint manager、evaluation output 和 local/manual script 边界。

## Goals / Non-Goals

**Goals:**

- 在 U-MaskBeamJEPA 缺失模态路径中新增 `fusion_type: pcpg`，第一版实现 logits/prototype-score 层融合，并记录可分析 gate 权重。
- 通过 training extension 增加 branch-balanced auxiliary CE、radar-protected CE、hard subset static weighting 和可选 JEPA latent alignment，避免扩写 trainer 主循环。
- 让 checkpoint selection 可显式选择 `val_acc`、`avg_missing_top1` 或 `worst_pattern_top1`，默认保持既有 early-stopping/top1 逻辑。
- 提供 oracle gate eval helper、6 组实验矩阵、并行 launcher、summary CSV/Markdown 和 focused tests。

**Non-Goals:**

- 不新增完整训练框架、旧式 root training script、package console script 或长期 public API。
- 不默认启用 PCPG、branch aux、radar protect、hard subset weighting 或 JEPA alignment。
- 第一版不强制实现 feature-level PCPG；若配置请求 `pcpg_fuse_level: features` 且当前路径不能低风险支持，应清晰降级或报错。
- 不提交真实训练输出、checkpoint、日志、generated config 或本地 dataset 内容。

## Decisions

1. **PCPG 放在 U-MaskBeamJEPA 内部，优先 logits 融合。**  
   该模型已经拥有每模态 latent、reliability、beam head、prototype bank 和 missing mask，是最小侵入位置。每个模态复用共享 `beam_head` / `prototype_bank` 得到 unimodal logits/prototype scores，再用 gate 对可用模态 logits 做 softmax 加权；不可用模态 logits 强制 mask 到 0 权重。备选方案是在 `ModularSequenceModel` 新增通用 representation core，但会扩大普通 baseline 的 forward surface，本轮不采用。

2. **辅助训练目标放在训练 extension。**  
   `BatchStepRunner` 已支持 extension `after_forward` 和 diagnostics，因此 branch-balanced/radar-protect/hard subset 作为新 extension 接入，trainer 只通过既有 `_build_training_extensions` 选择 extension。备选方案是直接修改 `_compute_base_loss`，但会污染普通 supervised 路径。

3. **Hard subset 第一版使用 loss reweighting。**  
   训练 batch 已经能拿到 missing mask / force mask 和 random-dropout metadata；静态 loss 权重可以在 extension 中根据 pattern 乘到基础 loss 或增加等价差值，不需要改 dataset sampler。sampler reweighting 后续可在有证据后单独实现。

4. **checkpoint selection 作为 checkpoint manager 的通用 helper。**  
   默认 `val_acc` 仍对应现有 top1 selection；新增 metric 只在 epoch log 中存在对应 missing metrics 时生效，否则清晰 fallback 或不更新。sidecar 记录 `selection_metric`、`selected_epoch` 和相关 task metrics。

5. **launcher/summarizer 保持 local/manual script。**  
   `scripts/launch_pcpg_radar_balance_v1.py` 只生成并发子进程、日志和 manifest；`scripts/summarize_pcpg_radar_balance_v1.py` 只读 outputs。二者不进入 pyproject console scripts，默认输出在 ignored `outputs/pcpg_radar_balance_v1/`。

## Risks / Trade-offs

- **PCPG 共享 beam head 的 unimodal logits 仍可能继承共享头偏置。** → 同时提供 branch aux / radar aux 和 oracle gate，分别验证训练保护与动态融合上限。
- **训练 extension 需要读取模型 diagnostics，字段可能缺失。** → extension 对缺失字段零损失/清晰诊断，普通模型不受影响；focused tests 使用 synthetic tensor 覆盖核心逻辑。
- **avg_missing/worst_pattern checkpoint metric 未必每轮 validation 都有。** → 默认不改变 existing selection；新增 selection 只在指标存在时更新，并在 sidecar/summary 标注实际 metric。
- **feature-level PCPG 改动面更大。** → 第一版只正式支持 `logits`，`features` 保留配置入口但不静默声称已实现。
- **新增 local/manual scripts 可能触发表面积护栏。** → 在 OpenSpec tasks 中登记 lifecycle、owner、输出边界和 dry-run test，并避免 package CLI 化。

## Migration Plan

1. 增加 opt-in config 字段和模型/extension/helper 实现，默认配置不变。
2. 增加 launcher、summary 和 tests。
3. 运行 OpenSpec validate、focused pytest 和脚本 dry-run/compile 验证。
4. 若需要回滚，删除新增 config flags、extension、PCPG helper、scripts/tests；既有训练默认行为不需要迁移。

## Open Questions

- radar prototype distance loss 是否能在第一版低风险接入正确 beam prototype；若不能，先落 radar auxiliary CE 并在 summary 中标注 prototype fallback。
- JEPA latent alignment 应优先复用现有 U-MaskBeamJEPA latent fields 还是现有 `loss.jepa` helper；实现时以当前模型输出可用字段为准。
