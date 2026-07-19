## ADDED Requirements

### Requirement: 训练初始化 checkpoint 必须与轨迹续跑分离
runtime MUST 提供仅加载模型权重并重置训练状态的显式 initialization checkpoint契约，并 MUST 与严格 `training.resume` 互斥。初始化 MUST 验证source SHA、checkpoint role/schema、加载key及新增missing key allowlist。

#### Scenario: 从成熟 Current checkpoint 初始化候选
- **WHEN** 候选配置声明 initialization checkpoint
- **THEN** runtime MUST 在optimizer构建前加载允许的expert与Current Router权重
- **AND** optimizer、scheduler、epoch、RNG、sampler和extension state MUST 从新run重新开始
- **AND** load report与source SHA MUST 写入checkpoint provenance

#### Scenario: 初始化身份不一致
- **WHEN** source SHA、shape、既有required key或unexpected key不符合声明
- **THEN** runtime MUST 在训练启动前失败

### Requirement: Router 校准必须冻结并固定 expert 运行状态
候选校准 MUST 将encoder、projection、reliability head、active/inactive beam head、temporal pooling和Current Router参数设为不可训练，并 MUST 在model进入train模式后继续令其BN/Dropout保持eval。optimizer MUST 只包含声明的候选 Router参数。

#### Scenario: 执行校准 optimizer step
- **WHEN** 一个候选batch完成backward和step
- **THEN** 只有candidate Router参数 MAY 变化
- **AND** frozen expert的参数与running statistics MUST 保持不变

### Requirement: 配对联合退化运行时必须传播固定 provenance
runtime MUST 使用内容寻址的240-entry Joint training panel，并 MUST 在run、checkpoint和summary中传播panel checksum、监督类型、source checkpoint和inner-only claim状态。

#### Scenario: 重复生成相同筛选任务
- **WHEN** seed、panel版本、source checkpoint和候选配置相同
- **THEN** resolved config与panel checksum MUST 相同
- **AND** train/evaluation corruption随机流 MUST 与其他seed或角色隔离
