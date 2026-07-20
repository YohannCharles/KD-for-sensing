## Context

PCER 快速验证已保存 A0-A3 的 resolved config、最佳 checkpoint、训练日志与 S0-S5 评测，但现有汇总只覆盖端到端结果和少量 router 均值。A1 在模态池化后路由，A3 在时间块原型证据上路由，二者权重粒度不同；因此诊断必须复用各自真实融合语义，同时保持样本、mask 和 checkpoint 身份一致。

## Goals / Non-Goals

**Goals:**

- 从验证集冻结统计 D1/D2 先验，在测试集用相同 backbone forward 离线重算 D0-D4 融合指标。
- 保留 time-major `[T,M] -> [N]` 语义，分别审计 A1 有效块系数和 A3 原生块权重。
- 定量检查 counterfactual target 的符号、KL 方向、分布、质量相关性、router 对齐和梯度路径。
- 对 S3 四个整模态缺失子场景输出指标、权重、误差和混淆，并比较逐块与整模态贡献。

**Non-Goals:**

- 不训练、微调或更新 A0-A3，不启动多 seed 或完整模态组合矩阵。
- 不改变历史 checkpoint、resolved config、快速验证结果或正式 claim。
- 不设计或实现下一代 router/target。

## Decisions

1. **单次 forward、多种融合重算。** A3 直接复用 block evidence logits；A1 复用模态特征和现有 prototype head 重算模态 evidence。D0-D4 只替换归一化权重，避免重复 backbone 推理和数据身份漂移。
2. **D1/D2 只由 validation split 估计。** 平均 raw router logits，并按每个位置实际可用次数归一；测试阶段先屏蔽不可用位置再 softmax。这样不会以测试集拟合静态替代先验。
3. **A0 prior 按真实静态语义映射。** A0 是均匀模态融合，不存在 learned scalar prior。A1 的 D3 与模态均匀 D4 相同；A3 的 D3 先给可用模态等质量，再在各模态可用时间块内均分，D4 则对所有可用块等权。
4. **A1 的块统计使用有效系数。** A1 router 只输出 `[B,M]`；将每个模态权重除以该模态可用时间块数，得到其 masked temporal mean 对应的有效 `[B,T,M]` 块系数，并在报告中明确这不是原生 block router 输出。
5. **梯度审计不调用 optimizer。** 对一个固定 batch 清零梯度，只对 `L_route` backward，记录 router 与非 router 参数梯度；随后丢弃模型实例。
6. **产物保持 local-only。** 源码只增加脚本和测试，诊断 CSV/JSON/Markdown 写入已忽略的 `outputs/`，结论标记为单 seed development evidence。

## Risks / Trade-offs

- [A1/A3 路由粒度不同，D1-D4 不能逐元素横比] → 分别按真实融合层定义，跨模型只比较最终指标和聚合模态权重。
- [固定 mask 多次评测仍有 I/O 成本] → 每个模型、场景只做一次 backbone forward，并在内存中更新所有替代模式统计。
- [训练日志未保存逐步梯度] → 以历史 epoch 日志结合单 batch 只读 backward，明确证据边界。
- [单 seed 快速验证波动未知] → 不机械宣称显著性，使用 0.2-0.3 个百分点仅作效应量参考。

## Migration Plan

无需迁移。新增入口可独立删除；历史训练、评测和 checkpoint 不受影响。

## Open Questions

无。诊断中的样本数、near-zero 阈值和 checkpoint 身份由运行产物显式记录。
