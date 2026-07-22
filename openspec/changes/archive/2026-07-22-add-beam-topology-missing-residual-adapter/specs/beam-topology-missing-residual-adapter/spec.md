## ADDED Requirements

### Requirement: residual adapter必须使用冻结四模态静态基础模型
系统 MUST 只消费 image、radar、gps、lidar 与64-beam index topology，并 MUST 冻结 backbone、prototype projector/bank、beam head和静态融合。B2没有合规已保存train-fit global mean或learned static prior时，系统 MUST 固定退回 C0 learned global prior，且R0-R5 MUST 共用同一checkpoint、config、prior和split。

#### Scenario: B2无法可靠静态化
- **WHEN** B2 checkpoint只含样本级动态Router且既有global mean来自validation统计
- **THEN** 系统 MUST 在manifest记录拒绝原因并使用C0-static
- **AND** 系统 MUST 不读取validation/test统计构造基础融合权重

### Requirement: paired cache必须无泄漏且可重建
系统 MUST 只从inner-train/inner-validation clean source保存sample identity、分层metadata、剩余模态evidence、full/missing logits和image/LiDAR residual；feature/logit量化后 MUST 通过full 0.999和两个missing 0.995的Top1 agreement gate。系统 MUST 拒绝outer test与channel、CSI、path、gain、power字段。

#### Scenario: 发布residual cache
- **WHEN** 预处理器完成train/validation cache转换
- **THEN** train/validation identity MUST 无交集且 residual MUST 满足 `z_minus + delta = z_full`
- **AND** 任一重建gate失败时系统 MUST 不启动adapter训练

### Requirement: adapter必须按精确缺失类型bypass
系统 MUST 只对恰好missing image或恰好missing LiDAR执行adapter。Full、missing radar、missing GPS、双模态缺失及其他mask MUST 原样返回基础logits，且adapter forward次数 MUST 为零。

#### Scenario: Full与负对照推理
- **WHEN** availability表示Full、仅缺radar或仅缺GPS
- **THEN** 输出 MUST 与输入基础logits逐元素一致
- **AND** image/LiDAR adapter均 MUST 不执行

#### Scenario: 目标单模态缺失推理
- **WHEN** availability恰好只缺image或只缺LiDAR
- **THEN** 系统 MUST 只拼接其余三个模态的64维evidence并输出64维residual
- **AND** 输出 MUST 等于基础missing logits加对应静态alpha乘residual

### Requirement: 六组实验必须共享训练与选择协议
系统 MUST 比较R0 no recovery、R1 train-mean、R2 modality-specific linear plain、R3 modality-specific linear topology、R4 shared mask-conditioned linear topology和R5 modality-specific MLP topology。R2-R5 MUST 共享seed、sample identity、train-only normalization/calibration、batch order、optimizer、epoch与只按inner-validation total loss选择checkpoint的规则。

#### Scenario: 训练learned adapter
- **WHEN** R2-R5开始训练
- **THEN** 只有adapter和modality-static sigmoid alpha MAY requires_grad
- **AND** teacher logits与residual target MUST stop-gradient

#### Scenario: 比较plain与topology objective
- **WHEN** R2与R3使用相同modality-specific linear结构
- **THEN** 两者结构、初始化与训练协议 MUST 相同
- **AND** 唯一方法差异 MUST 是R3增加仅依赖beam index topology的loss项

### Requirement: 报告必须覆盖恢复机制与停止门槛
系统 MUST 在同一inner-validation样本上报告Full、四种single-missing、S3 macro/worst、Top1/3/5、Within-3、MAE、oracle recovery、天气、8-sector、错误距离、D0-D3替换、残差相关性/动态性和效率。报告 MUST 判定R3六项success gates并给出唯一推荐。

#### Scenario: 汇总六组结果
- **WHEN** R0-R5结果齐全
- **THEN** 系统 MUST 校验所有方法的Full与非目标missing logits等于R0
- **AND** 系统 MUST 生成主表、恢复表、动态替换表和唯一继续或停止建议
- **AND** 系统 MUST 不自动启动outer test、multi-seed或下一轮训练
