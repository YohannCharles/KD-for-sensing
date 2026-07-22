## ADDED Requirements

### Requirement: Probe 必须只消费 inner clean 四模态 cache
系统 MUST 只读取 PR-SQDF cache 的 inner-train 与 inner-validation clean sample、beam index label、C0 block logits、global prior、availability、weather、scene和 full fused logits。系统 MUST 拒绝 outer/development/eval split、corrupted view、channel、CSI、path、gain、beam-power字段，且 MUST 不将 label、weather或scene放入模型输入。

#### Scenario: 加载 probe cache
- **WHEN** runner 读取 cache manifest和 clean shard
- **THEN** train/validation sample identity MUST 无重复且互斥
- **AND** input schema MUST 只含剩余模态 evidence
- **AND** 任一 forbidden字段或 outer-test标记 MUST 阻止运行

### Requirement: Modality evidence 必须等价复用 C0 时间 prior
系统 MUST 使用 C0 fixed prior和 availability mask聚合每个模态的五个时间 block，且 MUST 不自行平均。full logits、missing logits、evidence oracle和 residual oracle MUST 由同一融合实现构造。

#### Scenario: 构造四种 missing target
- **WHEN** 一个 clean sample分别移除 image、lidar、radar和gps
- **THEN** 每个输入 MUST 只拼接其余三个 64维 modality evidence
- **AND** `Z_minus_m + (Z_full - Z_minus_m)` MUST 数值等于 `Z_full`
- **AND** 真实 missing evidence按 full-observation prior加回 MUST 数值等于 cache-reconstructed Full

### Requirement: Cache reconstruction 必须在 probe 前通过 gate
系统 MUST 在全部 inner-validation cache报告 full reconstruction的 logits max/mean absolute error、Top1、Top3和Within-3 agreement，并 MUST 在固定 inner-validation原模型子集报告四种 missing reconstruction同组指标。full Top1 agreement MUST 不低于0.999，且每个 missing方向 Top1 agreement MUST 不低于0.995。

#### Scenario: 重建未达阈值
- **WHEN** full或任一 missing Top1 agreement低于阈值
- **THEN** 系统 MUST 保存失败报告
- **AND** 系统 MUST 不启动16个 probe任务或发布 recovery checkpoint

### Requirement: 轻量 probe 矩阵和训练身份必须固定
系统 MUST 对四个 missing modality分别训练 Linear Evidence、MLP Evidence、Linear Residual和MLP Residual，共16个任务。所有任务 MUST 共享 seed、split identity、train-fit normalization、batch order、epoch、optimizer与 early-stopping规则；checkpoint MUST 只按最低有限 inner-validation recovery loss选择。

#### Scenario: 训练一个 missing方向
- **WHEN** worker依次训练四个 probe
- **THEN** Linear MUST 是单层线性映射且 MLP MUST 使用预注册小型结构
- **AND** normalization统计 MUST 只从 inner-train拟合
- **AND** validation Top1、weather或sector结果 MUST 不参与 checkpoint选择

### Requirement: Evidence 和 residual loss 必须保持目标定义
Evidence probe MUST 使用 SmoothL1、distribution KL和 beam-topology distribution loss；Residual probe MUST 使用 SmoothL1、corrected-logit CE和现有 topology-aware beam loss。beam label MAY 作为 residual监督，但 MUST 不作为任何 probe输入。

#### Scenario: 计算 recovery objective
- **WHEN** evidence或residual batch执行 forward
- **THEN** evidence输出 MUST 以 missing modality evidence为目标
- **AND** residual输出 MUST 以 full-minus-missing logits为目标
- **AND** residual corrected logits MUST 等于 missing logits加 predicted residual

### Requirement: 基线、指标和分层必须完整报告
系统 MUST 对每个 missing方向统一报告 Full、No Recovery、Mean Evidence、NN Evidence、Oracle Evidence、Linear Evidence、MLP Evidence、Linear Residual和MLP Residual的 Top1、Top3、Top5、Within-3和MAE。系统 MUST 同时报告 target predictability、Top1/Within-3/MAE oracle-gap recovery、sunny/rainy/foggy和8个连续 beam sector。

#### Scenario: 汇总四个方向
- **WHEN** 四个 modality worker均完成
- **THEN** recovery与predictability CSV MUST 包含所有方法和方向
- **AND** gap recovery MUST 允许小于0或大于1且不得静默截断
- **AND** Top1 oracle gap小于0.5个百分点的方向 MUST 排除平均 gap-recovery判断但仍完整展示

### Requirement: Probe 完成后必须给出唯一建议并停止
系统 MUST 按预注册条件分别判断 evidence与residual可行性，回答四个缺失模态、oracle上限、线性/非线性收益、指标转化、LiDAR gap、mean/NN和weather一致性，并 MUST 只给出一个最终方向或停止建议。

#### Scenario: 生成 feasibility summary
- **WHEN** 16个任务和分层统计已完成
- **THEN** summary MUST 以 beam Top1、Within-3、MAE和oracle-gap recovery为主
- **AND** 高相关但无最终 beam改善 MUST 不得判为可行
- **AND** 系统 MUST 不自动启动 multi-seed、outer test或完整 fallback训练

### Requirement: 四 GPU launcher 必须隔离任务
launcher MUST 将 image、lidar、radar、gps分别映射到物理 GPU0、GPU1、GPU2、GPU3，并 MUST 在设置 `CUDA_VISIBLE_DEVICES` 后让进程内部使用 `cuda:0`。单个 worker失败 MUST 不发送信号给其他任务或无关GPU进程。

#### Scenario: 一个 modality worker失败
- **WHEN** 四个后台worker中一个以非零状态退出
- **THEN** 其他worker MUST 继续运行并被等待
- **AND** 汇总 MUST 报告缺失任务而不得伪造完整结论
