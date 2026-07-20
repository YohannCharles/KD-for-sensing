## ADDED Requirements

### Requirement: T2 动态可靠性候选必须共享单一组合契约
系统 MUST 以单一 `router_variant` 枚举表达 Current、PATR、H2R、CoRe 和 Unified-HPR，并 MUST 保持省略该字段时的 Current Router state-dict 与 forward 行为不变。系统 MUST 不为四个候选复制模型或训练循环。

#### Scenario: 加载 canonical T2
- **WHEN** canonical T2、S1 或 DeepSense6G T2 未声明候选 Router
- **THEN** 系统 MUST 构建 Current Router
- **AND** 不得实例化候选专用参数

#### Scenario: 选择候选方法族
- **WHEN** inner-only筛选配置声明 PATR、H2R、CoRe 或 Unified-HPR
- **THEN** 系统 MUST 按固定映射启用对应证据与门控组件
- **AND** 未知枚举或不兼容的 classifier/uniform 配置 MUST 被拒绝

### Requirement: PATR 必须区分静态模态能力与动态质量残差
PATR MUST 从逐帧 prototype 分布和有效帧统计构造时序质量证据，并 MUST 将最终 gate logits表示为训练集全局 prior与有界样本残差之和。动态残差 MUST 不接收 modality one-hot。

#### Scenario: 动态证据不可用
- **WHEN** 残差末层保持零初始化或样本只有一个可用模态
- **THEN** 路由权重 MUST 精确回退为 availability-normalized train-fit prior

### Requirement: H2R 必须在时序池化前门控坏帧
H2R MUST 先在每个模态内部计算 `[B,T,M]` 有效时间块权重，再计算 `[B,M]` 模态权重。无效时间块的权重 MUST 为零，最终融合 logits MUST 可由健康度池化后的 unimodal logits 和模态权重重构。

#### Scenario: 同一模态只有部分帧退化
- **WHEN** 模态仍可用但部分有效时间块的质量证据下降
- **THEN** 帧级健康门控 MUST 在模态池化前改变这些时间块的贡献
- **AND** 对应参数 MUST 从配对训练 loss 获得有限非零梯度

### Requirement: CoRe 必须使用无固定教师的跨模态共识
CoRe MUST 以每个模态之外的可用模态构造 leave-one-out prototype 共识，并计算分布分歧、topology距离及Top-k重合证据。系统 MUST 不指定固定传感器教师。

#### Scenario: 可用模态产生离群预测
- **WHEN** 一个可用模态高置信但与其他可用模态的 prototype 分布显著冲突
- **THEN** CoRe MUST 产生不同于一致模态的分歧证据

#### Scenario: 只有一个模态可用
- **WHEN** 样本只有一个可用模态
- **THEN** 共识证据 MUST 安全置零
- **AND** 唯一模态的最终融合权重 MUST 为一

### Requirement: 配对反事实训练不得泄漏退化元数据
系统 MUST 以相同样本和相同 availability 构造 drop-control 与 joint-corrupt view。state matrix、corruption名称、严重度和condition id MUST 仅用于数据变换与审计，不得进入模型 forward 或 Router loss。

#### Scenario: 构造联合缺失退化对
- **WHEN** 固定 panel 指定部分cell为Drop、部分cell为Corrupt
- **THEN** control与joint view的 availability MUST 完全相同
- **AND** 两者只能在Corrupt cell的传感器值上不同

#### Scenario: 计算相对质量约束
- **WHEN** control与joint的 detached单模态utility差超过预注册阈值
- **THEN** loss MUST 直接由utility差决定active模态和margin
- **AND** 不得读取affected-modality标签

### Requirement: Router 必须支持两种互斥连续效用监督
系统 MUST 支持 `label_topology` 和 `beam_power` 两种互斥监督。前者 MUST 使用 active BPA topology构造连续Gaussian效用；后者 MUST 使用训练期完整beam-power向量的归一化期望效用。

#### Scenario: 标签拓扑监督
- **WHEN** 配置选择 `label_topology`
- **THEN** 训练 MUST 不要求future beam-power
- **AND** 0/63等端点关系 MUST 由声明的topology而非硬编码数值距离决定

#### Scenario: Beam-power监督
- **WHEN** 配置选择 `beam_power`
- **THEN** 缺失、非有限、负值或shape不匹配的power target MUST fail closed
- **AND** 线性power target MUST 在`float32`中完成逐样本归一化和期望效用计算，不得因AMP cast发生下溢
- **AND** power target MUST 不进入模型forward

### Requirement: seed1 筛选必须固定身份和证据边界
系统 MUST 为八个候选任务记录source checkpoint SHA、variant、supervision、seed、split、Joint panel checksum、resolved config、GPU、训练预算和claim eligibility。seed1结果 MUST 标记为inner-only。

#### Scenario: 启动八卡筛选
- **WHEN** 用户确认启动本轮候选
- **THEN** launcher MUST 一卡一任务生成PATR/H2R/CoRe/Unified-HPR的label/power配对矩阵
- **AND** 身份预检失败时不得启动任何训练

#### Scenario: 候选晋级
- **WHEN** 汇总Dynamic、train-fit prior、Current、Uniform和Oracle
- **THEN** 只有同时超过静态prior的主指标并满足Clean/Drop保护条件的候选才可进入多seed确认
