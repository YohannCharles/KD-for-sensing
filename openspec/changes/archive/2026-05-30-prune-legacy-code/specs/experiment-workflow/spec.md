## MODIFIED Requirements

### Requirement: 推荐实验文档保持精简入口
实验工作流文档 MUST 将 README 作为入口地图，而不是完整实验手册。README MUST 指向 canonical config、docs 和 OpenSpec；详细实验矩阵、分析流程和调参说明 MUST 放在 `docs/` 或对应 specs 中。已退役的 G2D、CRAF、MARF 和 Multimodal-NF 内容 MUST 从 README 推荐入口和实验矩阵中删除。

#### Scenario: README 提供最短可运行路径
- **WHEN** 新用户阅读 README
- **THEN** 用户 MUST 能找到安装命令、快速健康检查、训练/评估/预处理/manifest 导出入口和数据产物边界
- **AND** 用户 MUST 能通过链接进入当前保留能力的详细实验矩阵或 viewer 文档

#### Scenario: 长实验说明迁移到 docs
- **WHEN** README 中的某段内容主要描述当前保留的 CSI hardening、viewer、Raymobtime 或 MMW 详细实验流程
- **THEN** 该内容 MUST 迁移到对应 `docs/` 文件或 OpenSpec spec
- **AND** README MUST 保留简短摘要和链接

#### Scenario: 退役研究线文档删除
- **WHEN** README、docs 或实验矩阵提到 G2D、CRAF、MARF 或 Multimodal-NF 推荐运行命令
- **THEN** 这些段落 MUST 被删除或改为明确说明该入口已退役
- **AND** 文档 MUST 不再推荐运行对应配置、测试或日志分析流程

### Requirement: 表面积收敛保持实验 artifact 兼容
删除冗余配置、入口或文档后，当前保留的训练和评估 workflow MUST 继续保存完整运行 artifact。使用保留的 virtual/overlay 配置时，运行目录 MUST 记录足够信息用于复现，不得要求用户恢复已删除的实体 YAML。已退役的 CRAF、MARF、G2D 和 Multimodal-NF 配置不得由 virtual alias 接管。

#### Scenario: virtual 配置训练 artifact 完整
- **WHEN** 用户使用当前保留的 virtual/overlay 配置启动训练并完成 artifact 写出
- **THEN** 运行目录 MUST 包含完整 `final_config.yaml`、`resolved_config.yaml`、`train_log.json`、checkpoint metadata 和 split/runtime metadata
- **AND** 这些 artifact MUST 足以说明实际模型、数据、loss、训练参数和 checkpoint 来源

#### Scenario: 删除 fallback 入口不影响 console script
- **WHEN** 重复脚本 wrapper 被删除
- **THEN** 对应 console script 或 `python -m kd_sensing.cli.*` 入口 MUST 继续通过 `--help` 检查
- **AND** README 推荐命令 MUST 使用仍存在的入口

#### Scenario: 研究脚本不进入核心 workflow 兼容承诺
- **WHEN** 保留的研究脚本未声明为包内 CLI
- **THEN** 核心训练、评估、预处理和 manifest 导出 workflow MUST 不依赖该脚本
- **AND** 该脚本的输出产物 MUST 继续位于 `.gitignore` 覆盖路径或显式本地输出目录

#### Scenario: 退役配置不被兼容接管
- **WHEN** 用户引用已删除的 CRAF、MARF、G2D 或 Multimodal-NF 配置路径
- **THEN** 配置加载器 MUST 给出清晰缺失或退役错误
- **AND** 系统 MUST 不生成同名 virtual 配置

### Requirement: 删除实体配置后 workflow 必须可复现
当当前保留的实体 YAML 被 recipe/overlay 替代后，训练和评估 workflow MUST 继续保存足够的 resolved/final 配置、运行元数据和 checkpoint 来源信息，保证不恢复被删除 YAML 也能理解实际运行参数。已退役的 CRAF、MARF、G2D 和 Multimodal-NF 实体 YAML 删除后 MUST 不提供同名 recipe/overlay 兼容。

#### Scenario: virtual 配置训练记录完整
- **WHEN** 用户使用当前保留的 virtual/overlay 配置完成训练或 dry-run artifact 写出
- **THEN** 运行目录 MUST 包含完整 `final_config.yaml`、`resolved_config.yaml`、训练元数据和 checkpoint 来源信息
- **AND** 这些 artifact MUST 能说明实际模型、数据、loss、训练参数和输出 run name

#### Scenario: 删除 YAML 不影响评估入口
- **WHEN** 某个当前保留的实体 YAML 被删除但对应 virtual/overlay 配置仍被声明支持
- **THEN** `kd-sensing-evaluate --config <deleted-yaml-path>` MUST 通过配置加载器解析等价最终配置
- **AND** 如果该路径未被声明支持，系统 MUST 抛出清晰缺失配置错误

#### Scenario: 退役 YAML 不支持 virtual fallback
- **WHEN** 被删除 YAML 属于 CRAF、MARF、G2D 或 Multimodal-NF
- **THEN** 系统 MUST 将其视为不支持路径
- **AND** 系统 MUST 不为其提供 virtual fallback

## REMOVED Requirements

### Requirement: 训练流程支持 CRAF 输出适配
**Reason**: CRAF/MARF dict 输出只服务于已退役架构。
**Migration**: 当前保留模型继续使用通用三元组或已保留模型自己的输出适配。

#### Scenario: CRAF/MARF dict 输出适配删除
- **WHEN** 模型 forward 返回 CRAF/MARF 专属 dict 字段
- **THEN** 训练流程 MUST 不再把该格式作为支持契约
- **AND** 相关正向测试 MUST 被删除

### Requirement: 训练流程支持 CRAF 附加 loss
**Reason**: CRAF beam-aware、auxiliary 和 counterfactual gate loss 随 CRAF 删除。
**Migration**: 使用当前保留的 task loss 和 distillation loss。

#### Scenario: CRAF 附加 loss 不再组合
- **WHEN** 配置包含 CRAF 附加 loss 字段
- **THEN** 系统 MUST 不把这些字段加入训练总 loss
- **AND** 当前保留训练流程 MUST 不导入 CRAF loss helper

### Requirement: 评估流程支持 CRAF 输出
**Reason**: CRAF checkpoint 评估随模型退役，不再作为公开工作流。
**Migration**: 评估当前保留模型 checkpoint。

#### Scenario: CRAF checkpoint 不再评估
- **WHEN** 用户尝试评估 CRAF checkpoint 或配置
- **THEN** 系统 MUST 因模型类型不可用而失败
- **AND** 评估流程 MUST 不包含 CRAF counterfactual 跳过逻辑

### Requirement: CRAF 日志与运行产物
**Reason**: CRAF reliability、counterfactual 和 auxiliary loss 摘要只服务于退役训练路径。
**Migration**: 普通训练继续写出通用 `train_log.json`、`metrics.json` 和 TensorBoard 标量。

#### Scenario: CRAF 日志字段不再要求
- **WHEN** 当前保留训练完成一个 epoch
- **THEN** `train_log.json` MUST 不要求包含 CRAF 附加 loss 或 reliability 字段
- **AND** 缺少这些字段 MUST 不影响验收

### Requirement: CRAF smoke test 工作流
**Reason**: CRAF smoke test 覆盖已退役模型和训练路径。
**Migration**: 使用当前保留 fusion、CLI help 和架构边界测试。

#### Scenario: CRAF smoke test 删除
- **WHEN** 开发者运行 focused tests
- **THEN** 测试清单 MUST 不再要求运行 CRAF synthetic 短训练

### Requirement: CRAF 稳定化训练工作流
**Reason**: CRAF 稳定化训练字段和反事实流程已退役。
**Migration**: 无兼容迁移。

#### Scenario: CRAF 稳定化流程不可用
- **WHEN** 用户配置 CRAF warmup、CE-only counterfactual、ignore band 或 gate/loss schedule
- **THEN** 系统 MUST 不再执行这些训练逻辑

### Requirement: CRAF 稳定化实验矩阵
**Reason**: CRAF 消融矩阵属于退役研究线。
**Migration**: 当前推荐实验矩阵只保留仍支持的方法。

#### Scenario: CRAF 消融入口删除
- **WHEN** 用户查看推荐实验矩阵
- **THEN** 系统 MUST 不再列出 CRAF no-counterfactual、fixed prior 或 token transformer gate baseline 作为 CRAF 实验

### Requirement: Teacher-prior CRAF stage workflow
**Reason**: teacher-prior CRAF Stage 1/2/3 workflow 退役。
**Migration**: 单模态 teacher 训练可使用普通单模态配置；不再生成 CRAF registry 或 stage workflow。

#### Scenario: Stage workflow 不可用
- **WHEN** 用户加载 teacher-prior CRAF Stage 2 或 Stage 3 配置
- **THEN** 系统 MUST 拒绝该配置
- **AND** 训练入口 MUST 不执行 teacher encoder initialization 或 selective fine-tuning 策略

### Requirement: Teacher registry build command
**Reason**: teacher registry build command 只服务于 teacher-prior CRAF/MARF。
**Migration**: 不再生成 teacher reliability registry；普通 checkpoint 路径继续由配置指定。

#### Scenario: teacher registry 命令删除
- **WHEN** 用户查找 teacher registry 构建脚本或 CLI
- **THEN** 项目 MUST 不再将该命令作为支持入口

### Requirement: Teacher-prior CRAF optimizer 参数组
**Reason**: Stage 3 参数组只服务于退役 teacher-prior CRAF。
**Migration**: 当前保留训练的 optimizer 参数组由通用 optimizer 构建逻辑处理。

#### Scenario: Stage 3 参数组测试删除
- **WHEN** 开发者运行 optimizer focused tests
- **THEN** 测试 MUST 不再要求 strong/weak encoder CRAF 参数组存在

### Requirement: Teacher-prior CRAF validation subsets
**Reason**: teacher prior 驱动的 subset 验证与 CRAF/MARF 退役架构绑定。
**Migration**: 普通模态子集评估若仍保留，必须不依赖 teacher-prior CRAF/MARF 命名。

#### Scenario: CRAF/MARF subset 名称删除
- **WHEN** 配置请求 `top_prior`、`single_best_prior` 或 `random_with_top_prior` 等 CRAF/MARF subset 名称
- **THEN** 系统 MUST 不再将这些名称作为 teacher-prior CRAF/MARF 验证入口

### Requirement: Teacher-prior CRAF smoke tests
**Reason**: teacher-prior CRAF 短训练和 gate 初始化测试覆盖已退役能力。
**Migration**: 删除对应测试；保留当前支持模型的 smoke。

#### Scenario: teacher-prior CRAF 测试删除
- **WHEN** 开发者运行快速回归
- **THEN** 测试 MUST 不再要求 PriorResidualGate 或 Stage 2/3 synthetic smoke 通过

### Requirement: G2D training workflow
**Reason**: G2D 训练 workflow 退役。
**Migration**: 使用普通 fusion no-KD、logits KD 或 RKD 训练。

#### Scenario: G2D 训练入口不可用
- **WHEN** 用户运行 G2D 配置
- **THEN** 训练入口 MUST 拒绝该配置
- **AND** 系统 MUST 不构建 G2D teacher ensemble、SMP 或 diagnostics

### Requirement: G2D validation commands
**Reason**: G2D focused tests 和 smoke training 随 G2D 退役。
**Migration**: 快速回归改为覆盖当前保留的核心 workflow。

#### Scenario: G2D 验证命令删除
- **WHEN** 开发者查看验证说明
- **THEN** 推荐命令 MUST 不再包含 `tests/test_g2d_loss.py`、`tests/test_g2d_distiller.py`、`tests/test_g2d_smp.py` 或 `tests/test_g2d_diagnostics.py`

### Requirement: Fusion 实验配置命名保持场景中立
**Reason**: 该 requirement 的场景主要服务 MARF/CRAF 高级配置命名；相关配置已删除。
**Migration**: 当前保留配置的命名约束由对应 canonical/experiment workflow 要求覆盖。

#### Scenario: MARF/CRAF 命名约束删除
- **WHEN** 开发者查看 fusion 配置命名要求
- **THEN** active specs MUST 不再要求 MARF 或 CRAF 配置路径存在

### Requirement: Multimodal-NF 运行产物一致性
**Reason**: Multimodal-NF dataset、objective 和 runtime metadata 退役。
**Migration**: 当前保留数据集的运行产物一致性由各自 specs 约束。

#### Scenario: Multimodal-NF runtime artifact 不再要求
- **WHEN** 当前保留训练或评估写出运行产物
- **THEN** 系统 MUST 不要求包含 Multimodal-NF dataset type、codebook metadata 或 enabled heads 一致性字段
