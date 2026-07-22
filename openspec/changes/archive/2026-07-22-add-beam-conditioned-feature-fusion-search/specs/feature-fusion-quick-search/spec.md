## ADDED Requirements

### Requirement: Frozen cache 必须保持 C0 身份与禁止输入边界
系统 MUST 从同一 C0 validation-best checkpoint 的 inner train、inner validation 和 development evaluation split 构建六个互斥 shard，并记录 checkpoint/config SHA、样本身份、split、四模态五时间步 pre-prototype feature、block logits、availability、modality/time id、static prior、base full logits 和非输入分组元数据。系统 MUST NOT 缓存或消费 channel、CSI、gain、beam power、ray/path、历史 beam、weather 或 scene 作为模型输入。

#### Scenario: 合并六个 cache shard
- **WHEN** 六个 shard 完成并执行 merge
- **THEN** 系统 MUST 拒绝重复、遗漏、split 混用、shape/dtype 不一致、源 SHA 不一致或禁止字段
- **AND** manifest MUST 标记 outer test 为 false、claim eligible 为 false

### Requirement: F0 parity gate 必须先于任何训练
系统 MUST 使用缓存 block logits、static prior 和 availability 重建 masked-softmax late logit sum，并与同一 frozen C0 在线输出比较 max/mean absolute error、Top1、Top3 和 Within-3 agreement。Full Top1 agreement MUST 不低于 0.999，四个 single-missing Top1 agreement MUST 分别不低于 0.995；未满足时系统 MUST 拒绝启动 F1--F5。

#### Scenario: 缓存精度不足
- **WHEN** 任一 Full 或 single-missing parity 指标低于门槛
- **THEN** preflight MUST 失败并保存具体 split、mask、误差和 agreement
- **AND** launcher MUST 不启动训练任务

### Requirement: F0--F5 必须共享公平融合输入与结构边界
系统 MUST 以相同 cached features、modality adapters、normalization、modality/time embedding 和 availability 语义构建 F1--F5。F1 MUST 使用 masked concat MLP 与冻结 prototype head；F2 MUST 使用 masked Transformer fusion token 与同一 prototype head；F3 MUST 使用 64 learned queries；F4/F5 MUST 使用 detached frozen prototypes 的共享投影作为 64 queries；F3--F5 MUST 使用同形 decoder 和 shared scalar score。F2--F5 MUST NOT 以 alpha 加权 block logits 作为最终输出。

#### Scenario: 构建六个方向
- **WHEN** preflight 为 F0--F5 生成 synthetic batch
- **THEN** 每个输出 MUST 为 `[B,64]` 且 F3--F5 query 数 MUST 为 64
- **AND** F4/F5 prototype 参数 MUST 冻结，F3/F4 除 query 来源外结构 MUST 相同

### Requirement: Missing token 必须从 feature interaction 中完全排除
系统 MUST 在 token 构建、attention key padding、pooling、auxiliary loss 和诊断中排除 unavailable block，且每个样本 MUST 至少存在一个 available block。系统 MUST 确保被 mask token 的数值变化不改变输出，并将 attention missing leakage 保持在数值容差内。

#### Scenario: 修改 missing token 内容
- **WHEN** 测试只随机替换 unavailable token 而保持 availability 不变
- **THEN** F1--F5 输出 MUST 在数值容差内不变
- **AND** F2--F5 对 unavailable key 的 attention MUST 为零或数值近似零

#### Scenario: all-missing 输入
- **WHEN** 任一样本的 20 个 block 全部 unavailable
- **THEN** 模型 MUST 在 forward 前给出明确错误

### Requirement: 训练预算与 checkpoint 选择必须预注册
F1--F5 MUST 使用同一 seed、inner train/validation、batch order、AdamW、scheduler、epoch、batch size、early stopping、structured mask schedule 和主 topology loss。F5 只可额外使用可用模态 auxiliary evidence loss。新增 lambda MUST 只用固定 train batches 校准一次；checkpoint MUST 仅由 Full 与 fixed masked-macro validation loss 等权组合选择，最终 Top1、Missing LiDAR 和 Worst MUST NOT 参与选择。

#### Scenario: F5 balanced evidence 训练
- **WHEN** F5 batch 含部分缺失模态
- **THEN** auxiliary loss MUST 只汇总当前可用模态并保持 prototype bank frozen
- **AND** 主 mask 分布 MUST 与 F1--F4 相同

### Requirement: 统一评测必须覆盖缺失、依赖性与 beam-specific interaction
每个 validation-best checkpoint MUST 在同一 development evaluation evidence 上报告 Full、四个 single-missing、S0--S5、Top1/3/5、Within-3、MAE、weather 和 8-sector 指标。系统 MUST 计算每模态 missing drop 与跨 sample token shuffle drop。F2--F5 MUST 报告 attention entropy 和 missing leakage；F3--F5 MUST 报告 beam-query pairwise JS、beam variance、neighbor/far similarity和 topology Spearman，且 MUST NOT 将 attention 称为 reliability。

#### Scenario: Prototype query 退化
- **WHEN** F4/F5 的 query attention pairwise JS 与 beam variance 接近零
- **THEN** 报告 MUST 将其判定为近似全局 pooling
- **AND** 系统 MUST NOT 声称 beam-conditioned interaction 成立

### Requirement: GPU0--5 任务与产物必须独立可审计
系统 MUST 固定 F0--F5 到 GPU0--5，启动前记录 GPU 状态，为每个任务保存 PID、resolved config、日志、状态、checkpoint 路径、metrics、weather/sector 和 efficiency。单任务失败 MUST NOT 终止其他任务；系统 MUST NOT 杀死其他进程、静默调参或自动重跑失败任务。

#### Scenario: 一个方向训练失败
- **WHEN** 任一方向以非零状态退出
- **THEN** launcher MUST 等待其他方向并记录该方向退出码与错误状态
- **AND** 汇总 MUST 将其标记为 failed 而不是伪造缺失指标

### Requirement: 筛选结论必须遵守 inner-only 停止边界
系统 MUST 按预注册 Full、Missing LiDAR、Missing Avg、Worst、Within-3、MAE、beam-specificity 和成本阈值比较 F0/F1/F2、F2/F3/F4 与 F4/F5，并给出唯一推荐方向。结果 MUST 标记 single-seed、inner/development、claim-ineligible；完成后 MUST NOT 自动运行 outer test、multi-seed、解冻 encoder 或下一轮端到端训练。

#### Scenario: 生成最终 comparison
- **WHEN** 可用方向的统一评测与诊断完成
- **THEN** 报告 MUST 明确回答融合层级、query 来源、LiDAR 依赖、天气/sector 一致性、参数与延迟问题
- **AND** workflow MUST 在唯一推荐方向后停止
