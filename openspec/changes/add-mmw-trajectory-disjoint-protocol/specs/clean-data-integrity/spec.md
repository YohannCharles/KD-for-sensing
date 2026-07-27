## MODIFIED Requirements

### Requirement: MMW 只能使用已绑定的正式 protocol

MMW 训练和开发验证 MUST 通过 `mmw_clean_inner_development_v1` 或 `mmw_trajectory_disjoint_v1` 中一个精确绑定且审计通过的 protocol 建立数据域。配置 MUST 显式绑定 protocol id、fingerprint、审计报告、train role、validation role，并显式关闭未授权 test 与 confirmation training。trajectory-disjoint MUST 是普通模型研发主协议；clean-inner 仅保留为旧协议比较和既有 canonical workflow。

#### Scenario: 构建合法 MMW loader

- **WHEN** 配置与受支持 protocol、审计报告和域清单完全一致
- **THEN** 系统 MUST 仅构建 train 与 validation loader
- **AND** 每个 loader 的运行元数据 MUST 记录 protocol 与 audit identity

#### Scenario: 使用未绑定或历史 MMW 配置

- **WHEN** MMW 配置缺少 protocol identity、审计报告、role 或 test 禁止字段
- **THEN** 系统 MUST 在创建 dataset、optimizer 或 checkpoint 前失败

### Requirement: train、validation 与封存 test 必须完全隔离

协议校验 MUST 拒绝 confirmation、trainval、merged train/validation 训练路径。clean-inner MUST 对每个 train domain 与 validation domain 审计 sample id、target id、完整行、原始资源、时间窗口和目标帧；trajectory-disjoint MUST 进一步对 train/validation/test 两两审计完整 trajectory group、scenario execution、依赖帧和四模态/channel 审计资源。任一要求隔离的重叠 MUST 失败。

#### Scenario: 非训练 split 被并入训练输入

- **WHEN** train 与 validation/test 共享要求隔离的身份或资源，或 validation/test 被加入训练
- **THEN** 审计 MUST 失败
- **AND** 配置不得进入训练或开发评估

### Requirement: 可拟合状态只能来自训练集

GPS scaler 与其他 normalization artifact MUST 只由所绑定 protocol 的 train loader 拟合。validation/test MUST 不参与 optimizer、scheduler、extension state、采样、prototype 初始化、class prior、checkpoint selection 以外的状态更新或任何可拟合统计更新；checkpoint selection 只能读取 validation loss。

#### Scenario: 运行开发验证

- **WHEN** 系统对 MMW validation loader 评估
- **THEN** validation MUST 是只读输入
- **AND** 未授权 test MUST 保持未访问

