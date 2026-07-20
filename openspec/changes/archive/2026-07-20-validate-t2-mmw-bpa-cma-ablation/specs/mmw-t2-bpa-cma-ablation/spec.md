## ADDED Requirements

### Requirement: 系统必须提供可归因的 T2 消融方法
系统 SHALL 提供 `T2`、`T2-NoBPA`、`T2-BPA2CMA`、`T2-Linear`、`T2-CLS` 和 `T2-CLS-CMA` 六个定义明确的方法行，并保证每一对因果比较只改变其声明的目标、拓扑或 prototype package。

#### Scenario: 仅关闭 BPA 辅助目标
- **WHEN** 生成 `T2-NoBPA` 配置
- **THEN** 系统将融合和单模态 BPA auxiliary loss 关闭，同时保留 T2 的 prototype head、prototype bank、router prototype-margin、superset KL、数据协议和训练预算

#### Scenario: 将 BPA 替换为 CMA
- **WHEN** 生成 `T2-BPA2CMA` 配置
- **THEN** 系统关闭 BPA auxiliary loss、启用 AMBER 风格 batchwise CMA analogue，并保持 prototype head、router 和其余 T2 配置与 `T2-NoBPA` 一致

#### Scenario: 完整去 prototype package
- **WHEN** 生成 `T2-CLS` 或 `T2-CLS-CMA` 配置
- **THEN** 系统使用 classifier head 并关闭 prototype alignment、modality prototype alignment 和 prototype-margin，且两者之间只允许 CMA 开关不同

### Requirement: CMA 必须遵循跨 batch 样本配对语义
系统 MUST 将每个可用模态特征作为 anchor，将同一样本融合特征作为正样本，将 batch 中其他样本融合特征作为负样本，并使用 cosine similarity、固定温度和可配置权重计算 InfoNCE 类辅助目标。

#### Scenario: batch 内其他样本参与分母
- **WHEN** 保持一个 anchor 及其正样本不变而修改其他样本的融合特征
- **THEN** CMA loss 随负样本相似度发生变化

#### Scenario: 缺失模态不作为 anchor
- **WHEN** availability mask 将某个模态标为缺失
- **THEN** 该模态特征不进入 CMA anchor 集合且不贡献 loss

#### Scenario: 重复样本身份使用多正样本
- **WHEN** 带放回 sampler 使同一稳定样本身份在 batch 内出现多次
- **THEN** 所有同身份融合候选均进入正样本集合且不得彼此作为负样本

#### Scenario: CMA 不依赖 beam label
- **WHEN** 特征、availability 和样本身份保持不变而 beam label 被打乱
- **THEN** CMA loss 数值保持不变

#### Scenario: 缺少稳定样本身份时失败
- **WHEN** CMA 已启用但 batch 无法提供与 batch size 一致的非空稳定样本身份
- **THEN** 训练以明确错误停止而不是静默采用 diagonal-only 配对

### Requirement: BPA 与 CMA 主消融配置必须互斥
系统 MUST 默认关闭 CMA，并在同一主消融配置同时启用 BPA 与 CMA 时拒绝启动，以保证 `BPA2CMA` 表示目标替换。

#### Scenario: 默认配置不变
- **WHEN** 旧配置未声明任何 CMA 字段
- **THEN** 系统保持当前 T2 loss 行为且不计算 CMA

#### Scenario: 冲突配置被拒绝
- **WHEN** 配置同时启用 BPA 和 CMA analogue
- **THEN** 配置校验在训练前返回可操作的互斥错误

### Requirement: linear 消融必须只切换 BPA target 拓扑
系统 SHALL 让 `T2-Linear` 只将 BPA Gaussian target 从 circular distance 切换为 linear distance，并保持 router、评估 metric、prototype head、loss 权重、soft-target sigma、seed、数据和训练预算与完整 T2 一致。

#### Scenario: 生成 linear 配置
- **WHEN** launcher 生成 `T2-Linear` 配置
- **THEN** 产物 provenance 显式记录 prototype target 为 linear、router 与评估为 circular，并记录其他关键 T2 配置与完整行相同

### Requirement: 训练和评估必须使用固定 MMW 协议
系统 MUST 在 15 个场景天气域上对五个新增方法运行 seeds `1,2,3`、40 epochs，并使用与完整 T2 相同的 balanced sampler、缺失 curriculum、epoch-40 checkpoint、持久化评估 mask 和有效样本集合。

#### Scenario: 多 seed 训练计划
- **WHEN** 用户启动完整消融矩阵
- **THEN** launcher 生成 15 个互不覆盖的训练任务并允许显式分配至 GPU0-7

#### Scenario: 配对评估身份校验
- **WHEN** 汇总两个方法的 paired delta
- **THEN** 系统先验证 sample identity、mask identity、target 和有效样本计数完全一致，任一不一致都拒绝输出 paired claim

### Requirement: 评估必须提供全量与 beam 边界切片
系统 SHALL 同时报告全量指标、精确端点 beam `{0,63}`、近端点 beam `{62,63,0,1}` 和其余内部 beam 指标，以检验 circular 与 linear BPA target 差异是否来自码本索引边界。

#### Scenario: circular 与 linear 分层比较
- **WHEN** `T2` 与 `T2-Linear` 完成相同掩码评估
- **THEN** 汇总输出端点和内部切片上的 Exact、Within-1、Within-3、beam error 及 paired delta

### Requirement: 输出必须支持论文机制图且约束 claim
系统 SHALL 输出三 seed 均值/标准差、缺失率曲线、paired delta 和绘图数据，并明确 CMA 行属于池化特征上的 AMBER-style objective analogue。

#### Scenario: 生成论文消融产物
- **WHEN** 所有方法通过完整性和身份一致性校验
- **THEN** 系统生成机器可读汇总、Markdown 表格和机制图，且不把 CMA analogue 标为完整 AMBER Class-Former 复现

#### Scenario: 结果不支持预期机制
- **WHEN** 某消融差异不稳定、置信区间跨零或只在局部切片成立
- **THEN** 汇总如实保留负结果并限制 claim，不筛选 seed、样本或缺失率
