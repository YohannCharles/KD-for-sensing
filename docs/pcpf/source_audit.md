# PCPF-T 源架构审计

本审计对应 `add-pcpf-temporal-risk-fusion`，范围只包括当前 PCPF-T 主线及其直接依赖。U0、AMBER-Full、RMBP-MM、DeepSense6G、MMW、CSI/TSPC、数据与 cache 均保留。

## 现有契约

- U0 的四个 encoder 接受五帧输入；`_encode_sequence` 将二维 encoder 输出补成三维，并要求最终表示为 `[B,T,D]`。PCPF-T 使用同一 encoder registry，但构造独立模型，不继承 U0。
- 时间缺失真值是 `modality_temporal_mask=[B,T,4]`，其中 `true` 表示该帧该模态有效；`available_modalities=mask.any(dim=1)`。PCPF-T 的 T-CLS 永远可见，缺失帧进入 Transformer padding mask，整模态缺失后显式清零。
- `BeamPrototypeBank` 只持有一份 `[64,64]` 可学习 prototype，以归一化 feature/prototype 的 cosine logits工作。现有 `prototype_alignment_loss` 已覆盖 fused feature、逐模态 feature、availability mask 与 circular topology soft target，因此直接复用。
- U0 Router 保持原样，只属于 `u_mask_beam_jepa`。`pcpf_temporal_risk_fusion` 拥有独立 state dict；解析式 A4 不构造自由四维 Router，A2 control 才显式构造 `direct_router`。
- 通用 trainer 负责 dataloader、初始化 checkpoint、optimizer/scheduler、validation-best 与 checkpoint schema。PCPF-T 只通过 opt-in `prepare_training_stage` 和 training extension 接入，不复制 trainer。
- train-only preparation 只接收 train loader：Stage 2 拟合风险分量 mean/std 和各模态 confidence P90；Stage 3 拟合 `mean_train_risk`。正式配置遍历完整 train，只有明确的 bounded smoke 可限制 batch。
- 通用 15-mask evaluator 已提供固定 mask、Top-1/3/5、Within-3、circular MAE 与 ECE。PCPF 专用评估器在其数据/forward 契约上增加 risk、weight、替换融合、calibration、weather/domain 和 confident-but-wrong 诊断。
- MMW pooled metadata 提供 `condition` 与 `scenario`，可形成 sunny/rainy/foggy 和 15 个 domain。PCPF 模型 forward 禁止读取 weather、domain、CSI、channel、path、beam power 或历史 beam。
- CSI/TSPC 是保留的独立研究边界。PCPF 源码不导入、不初始化、不修改 CSI/TSPC owner；相关数据、cache 和 checkpoint 不参与本 change 的清理。

## PCPF-T 数据流

`四模态五帧输入 -> 各模态 encoder -> 共享 Temporal Transformer -> 单一 BeamPrototypeBank -> deterministic unimodal probability -> 共享概率嵌入 -> 四项拓扑风险 -> 每模态温度校准 -> availability-aware 解析权重 -> 概率融合`

风险项为 `U_var`、`U_proto`、circular `U_temp` 与 `U_conflict`。所有 risk、softmax、exp、log 与 KL 路径在 FP32 中执行；missing weight 为零，Single weight 为一，权重行和为一。

## 阶段与晋级

| 阶段 | 可训练参数 | 来源与约束 |
| --- | --- | --- |
| Stage 1 | encoder、projection、共享 Temporal Transformer、唯一 prototype bank | 从随机初始化开始；不训练动态权重 |
| Stage 2 | probability head、共享非负风险系数与 bias | 只接受 Stage 1 validation-best；train-only 拟合 normalization/P90 |
| Stage 3A | 每模态 temperature、tau、显式启用时 eta/风险系数 | 只接受 Stage 2 validation-best；必须绑定未截断且通过的 gate JSON、SHA256 和相同 expert fingerprint |
| Stage 3B | Stage 3 标量、probability/risk heads | 只接受 Stage 3A validation-best；沿用同一 Stage 2 gate 绑定 |

`tools/run_pcpf.py` 只负责 resolve、preflight、显式训练委托和 smoke；`tools/eval_pcpf.py` 只读取 inner train/validation。任何 bounded gate 自动失败，不会生成可晋级的 Stage 3 配置。
