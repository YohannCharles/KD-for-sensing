## ADDED Requirements

### Requirement: Checkpoint 反序列化必须显式区分信任级别
项目中的 state-dict、tensor checkpoint 和 artifact metadata loader MUST 显式使用安全加载模式。需要任意 pickle object 的 legacy loader MUST 要求显式 trusted-local opt-in，并 MUST NOT 接受远程或来源不明输入。

#### Scenario: 普通 state-dict checkpoint
- **WHEN** runtime 加载模型、optimizer 或统计 artifact 的 tensor/dict checkpoint
- **THEN** loader MUST 显式使用 `weights_only=True` 或等价安全模式
- **AND** 安全模式失败 MUST 给出 schema 错误，而不是自动回退到 unsafe pickle

#### Scenario: Legacy trusted-local 例外
- **WHEN** 受保护历史 artifact 确实需要任意 pickle object
- **THEN** 调用方 MUST 显式设置 trusted-local opt-in
- **AND** metadata 或 warning MUST 记录 unsafe 模式与来源路径
- **AND** 远程 URL、下载缓存或来源未知路径 MUST 被拒绝

### Requirement: 批量预处理不得静默部分成功
批量预处理 MUST 使用稳定资源 identity 避免 basename 碰撞，MUST 聚合有限失败样本，并 MUST 在零成功、碰撞或失败超过明确阈值时返回失败。最终 CSV、JSON 和 metadata MUST 使用原子写。

#### Scenario: 不同目录同名输入
- **WHEN** 两个输入资源 basename 相同但规范化相对路径不同
- **THEN** preprocessor MUST 为它们生成不同 identity 或明确报告冲突
- **AND** MUST NOT 静默覆盖同一输出

#### Scenario: 全部样本失败
- **WHEN** batch preprocessing 没有任何成功样本
- **THEN** command MUST 返回失败并报告有限错误示例和总计数
- **AND** MUST NOT 写出看似成功的空结果 artifact

#### Scenario: 原子结果写出
- **WHEN** preprocessor 完成 CSV、JSON 或 metadata 生成
- **THEN** output MUST 先写入同文件系统临时路径并原子替换目标
- **AND** 写入失败 MUST 保留原目标

#### Scenario: 内部验证异常
- **WHEN** config 或 label-space validation 遇到内部导入或编程错误
- **THEN** validation MUST 暴露异常并阻止 workflow
- **AND** MUST NOT 捕获宽泛异常后跳过验证
