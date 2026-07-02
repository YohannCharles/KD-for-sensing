## Context

V3 主线使用 `UMaskBeamJEPA` whole-model exception，但关闭 JEPA、KD、RBMA 和 full auxiliary loss，实际路径是 strong encoders + `weighted_sum` reliability fusion + missing mask + beam prototype alignment。现有 prototype alignment 已有 `BeamPrototypeBank`、soft beam target 和可用模态 mask，最小改动应沿用这些实现。

## Goals / Non-Goals

**Goals:**

- 用 `use_beam_topology_proto`、`proto_target_type`、`tau_beam` 和 `circular_beam_distance` 控制 BTAPA target。
- 让 fusion feature 必选参与 prototype loss，可用 modality feature 可配置参与，缺失 modality 不参与。
- 提供默认关闭的 ADBA-aware auxiliary prototype loss。
- 新增 Scene31 BTAPA 配置、smoke test、串行 launcher 和 V3-vs-BTAPA 分析脚本。

**Non-Goals:**

- 不替换旧 V3 配置，不启用 RBMA、JEPA、KD 或 full auxiliary loss。
- 不新增 `train.py` root 旧入口，不改评估 hard-label 指标语义。
- 不增加新依赖，不大规模重构训练循环。

## Decisions

- 复用 `BeamPrototypeBank`，在 `beam_prototype_alignment.py` 增量扩展 target 类型和 diagnostics。替代方案是新增完整 class；当前单调用路径不需要额外抽象。
- BTAPA 使用 `soft_y[k] = exp(-distance(k,y)^2 / tau_beam)` 并归一化；旧 onehot 和旧 Gaussian soft target 通过 `proto_target_type` 保留。
- `lambda_proto` 继续作为总 BTAPA 权重，`btapa_fusion_weight` 与 `btapa_modality_weight` 只控制 `L_btapa` 内部组合，避免改变训练扩展的外部 loss 结构。
- ADBA-aware loss 只对启用的 prototype logits 计算 near-beam probability，默认关闭并单独记录 `adba_proto_loss`。
- launcher 使用当前 `kd-sensing-train` CLI；用户示例里的 `python train.py` 以当前 CLI 作为等价命令满足，不恢复已删除 root 入口。

## Risks / Trade-offs

- [Risk] 旧配置字段 `beam_label_sigma/beam_label_circular` 与新字段语义相近但公式不同。→ Mitigation: 旧 V3 保持旧字段，新 BTAPA 配置显式声明 `proto_target_type: beam_soft` 和 `circular_beam_distance: false`。
- [Risk] 新 diagnostics 字段可能为空。→ Mitigation: loss 关闭时记录 0，`metrics.csv` 动态列写出保持旧 reader 兼容。
- [Risk] 本地输出 CSV 不存在时分析脚本无法给出完整结论。→ Mitigation: 输出空字段并打印 unavailable 结论，不伪造指标。
