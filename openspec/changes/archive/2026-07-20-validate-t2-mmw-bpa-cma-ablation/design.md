## Context

当前 MMW 主线已经有完整 T2、S1、AMBER-Full 和 RMBP-MM 的三 seed 结果。T2 的 BPA 由 64 个可学习 beam-class prototypes、基于 beam 距离的 Gaussian soft target，以及融合/单模态 prototype KL 组成；同一 prototype bank 还用于主分类 head，router 也可使用 prototype-margin。因此“关闭 BPA 辅助损失”和“完全移除 prototype”是两个不同干预。

AMBER 论文公式 (34) 的 CMA 以每个可用模态 class query 为 anchor、同一样本 fusion query 为正样本、batch 内其他样本 fusion query 为负样本。当前本地 AMBER-Full 适配实现使用同一样本内的模态维 softmax，不具备该跨 batch 语义；T2 现有 `lambda_supcon` 又是按 beam label 配对，二者均不能直接充当本次目标替换。

MMW 使用完整 64-beam ULA-DFT 码本的原始列顺序，索引 63、0、1 在阵列主值映射的 boresight 附近连续。Town03 配置中的 `120` 是 LiDAR range，不是基站 120 度覆盖字段；即使覆盖扇区有限，只要没有按角度裁剪并重编号，0/63 仍可相邻。因此 circular 是“DFT 码本索引先验”，不是物理覆盖 360 度的主张；linear 仍是必要反事实，且必须通过端点与内部 beam 分层判断差异来自哪里。

## Goals / Non-Goals

**Goals:**

- 在保持数据、模型宽度、训练预算、seed 和缺失协议不变时，分别识别 BPA 辅助目标、CMA 辅助目标、circular beam 拓扑及 prototype package 的贡献。
- 产生可审计的三 seed 结果、paired delta、端点切片和论文图表，使“为什么 T2 优于 baseline”的论证由受控实验支撑。
- 对 CMA 的正负样本、重复样本、多模态可用掩码和配置互斥关系提供单元测试。

**Non-Goals:**

- 不给 MMW 人工加噪，不修改数据 split，不改变 15 域定义。
- 不在本次 change 中重写完整 AMBER Class-Former 或追溯修改已完成 baseline checkpoint。
- 不把池化特征 CMA analogue 宣称为完整 AMBER 复现。
- 第一轮不做大范围 loss-weight sweep；只有 CMA 训练出现明显失稳时，才增加预先记录的单 seed 敏感性检查。

## Decisions

### 1. 使用六个互补方法行，复用完整 T2

实验矩阵包含：

- `T2`：现有完整 T2，直接复用三 seed checkpoint。
- `T2-NoBPA`：仅将融合与单模态 BPA auxiliary loss 权重置零；保留 prototype head、prototype bank、router prototype-margin 和其他 T2 组件。
- `T2-BPA2CMA`：在 `T2-NoBPA` 基础上加入 batchwise CMA analogue；这是 BPA 与 CMA 的纯辅助目标替换。
- `T2-Linear`：只将 BPA Gaussian target 的距离由 circular 改为 linear；router 和评估距离继续使用 circular，与完整 T2 的其余配置相同。
- `T2-CLS`：使用 classifier head，关闭 BPA、modality BPA 和 prototype-margin；这是完整去 prototype package 的对照。
- `T2-CLS-CMA`：在 `T2-CLS` 基础上加入相同 CMA analogue，用于判断去 prototype 后 CMA 是否仍提供收益。

没有 `T2-CLS` 就不能把 `T2-CLS-CMA` 的差异归因于 CMA；没有 `T2-NoBPA` 就不能把完整 T2 与 CMA 的差异归因于辅助目标。因此两组匹配对照都保留。

### 2. CMA 使用跨 batch、多正样本 InfoNCE

复用 T2 已有融合特征 `z_f:[B,D]`、模态特征 `z_m:[B,M,D]` 和 availability mask，不新增 projection head。每个可用 `(sample, modality)` 是 anchor，batch 中融合特征为候选；与 anchor 具有相同稳定样本身份的所有候选均为正样本，其余为负样本：

`L = mean(logsumexp(all candidates) - logsumexp(same-identity candidates))`。

这样既遵循 AMBER 公式的跨 batch 语义，又避免 domain-balanced sampler 带放回时把同一 `sample_id` 的重复项错误当作负样本。启用 CMA 时若无法构造与 batch 等长的稳定身份，训练必须快速失败，不能静默退化为可能错误的 diagonal 配对。

温度固定为 `0.2`。主消融权重固定为 `0.2`，与 AMBER 论文给出的 `lambda_c=0.2` 及 T2 BPA 外层权重一致；训练日志必须同时记录未加权和加权 CMA，便于检查 loss 尺度。若首个 seed 出现非有限值或辅助项长期压倒主 CE，才允许增加 `0.05` 的单 seed 敏感性行，且不得用该检查替换预注册主行。

### 3. CMA 与 BPA 在主消融中互斥

配置解析必须默认关闭 CMA，并拒绝 `use_beam_prototype_alignment=true` 与 `use_amber_cma_analogue=true` 同时启用。该约束保证 `BPA2CMA` 是目标替换而非两个正则项叠加。CMA 不读取 beam label，因此打乱 beam label 不应改变其数值。

### 4. circular/linear 只改变 BPA target 的 wrap prior

`T2-Linear` 必须保持 prototype head、BPA 权重、soft-target sigma、训练 seed、router 监督和评估 metric 不变，只通过独立配置将 BPA Gaussian target 的距离从 `min(|i-j|, 64-|i-j|)` 改为 `|i-j|`。现有全局 `circular_beam_distance` 还会影响 router，不能用于这个单因素消融。汇总除全量指标外，还报告精确端点 `{0,63}`、近端点 `{62,63,0,1}` 与其余内部 beam，重点比较 `T2-Linear - T2` 的 Top1、Within-1/3 和 circular error。

### 5. 固定训练与评估协议

五个新方法各运行 seeds `1,2,3`，40 epochs，使用现有 MMW 15 域 balanced sampler、相同 curriculum 和 epoch 40 `last.pth`。完整 T2 复用已完成 checkpoint。训练按 GPU0-7 两波并行；输出目录含方法与 seed，避免覆盖现有主实验。

task-output 主消融使用同一组持久化 mask identities，覆盖 clean 和随机模态时间块缺失 `20/40/60/80%`；这是本轮预注册机制曲线。既有 `85/90/95%` extreme matrix 可在主结论稳定后作为二级压力测试复用，但不进入本轮必需的端点机制图。汇总必须先校验样本身份、mask 身份、target 和有效样本计数完全一致，再计算 paired delta。

### 6. 结果图服务于机制而非装饰

最终至少输出：

- 六方法在 Clean、Drop80 和 missing-AUC 上的三 seed 均值和标准差图；
- `T2`、`T2-NoBPA`、`T2-BPA2CMA` 的缺失率曲线，隔离辅助目标贡献；
- `T2` 与 `T2-Linear` 的精确端点/近端点/内部 beam paired-delta 图，判断 circular BPA target 收益来源；
- `T2-CLS` 与 `T2-CLS-CMA` 配对图，判断 CMA 在无 prototype package 时的净贡献。

若某项没有稳定提升，按负结果报告，不通过挑选 seed、样本或缺失率制造视觉优势。

## Risks / Trade-offs

- [CMA analogue 缺少 AMBER class-query cross-attention] -> 明确限定为 objective-level analogue，并保留完整 AMBER-Full baseline 作为 package comparison。
- [CMA loss 数量级随 batch 候选数变化] -> 固定 batch protocol，记录 raw/weighted loss；只有预注册的失稳条件触发 `0.05` 敏感性检查。
- [重复 `sample_id` 或跨域身份碰撞污染负样本] -> 使用数据集提供的稳定、域限定身份并做 batch 长度/非空校验；重复身份统一视为多正样本。
- [旧全局 circular 开关同时影响 BPA 与 router] -> 新增默认继承旧行为的独立 `prototype_target_circular` 配置；本次只切该字段，router 与评估保持 circular。
- [15 个新训练耗时] -> 复用完整 T2，GPU0-7 分波并行；先完成四种配置 smoke test，再提交全量训练。
- [外部 GPU 负载导致运行时间波动] -> 记录启动、结束、GPU 和 checkpoint 完整性，不以 wall-clock 吞吐作为模型结论。

## Migration Plan

1. 新增默认关闭的 CMA loss 和配置字段，先完成单元测试与配置互斥测试。
2. 扩展 MMW launcher/evaluator，使新方法仅通过显式 `--methods` 进入，不改变现有默认主实验。
3. smoke run 验证六方法配置差异与一个 optimizer step；随后并行运行三 seed。
4. 使用固定 checkpoint 和 mask artifacts 评估、提取 task outputs、生成汇总与图。
5. 若需回滚，只删除新增方法配置和可选 loss 路径；默认 T2、AMBER-Full 及其已有 artifacts 不受影响。

## Open Questions

- MMW 数据加载后的稳定样本身份是否已经包含完整域信息，还是需要由 `domain_id + sample_id` 组合；实现前通过实际 batch 审计确认。
- circular 优势若只出现在端点而总体置信区间跨零，论文应将其表述为码本边界鲁棒性而非总体精度创新点。
