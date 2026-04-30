## ADDED Requirements

### Requirement: 虚拟 canonical 配置工作流
训练、评估和测试工作流 MUST 接受由配置加载器生成的虚拟 canonical fusion 配置。虚拟配置 MUST 在进入训练、评估、dry-run、override 合并、验证和 artifact 写出之前被解析为完整配置字典。

#### Scenario: 训练入口使用虚拟 canonical 配置
- **WHEN** 用户运行 `python scripts/train.py --config configs/fusion/gps_mmwave_logits_kd.yaml`
- **THEN** 系统 MUST 解析该 canonical path 并启动 fusion logits KD 训练流程
- **AND** 训练流程 MUST 不要求 `configs/fusion/gps_mmwave_logits_kd.yaml` 在磁盘上存在

#### Scenario: 评估入口使用虚拟 canonical 配置
- **WHEN** 用户运行 `python scripts/evaluate.py --config configs/fusion/gps_mmwave_logits_kd.yaml --weights <path>`
- **THEN** 系统 MUST 解析该 canonical path 并构建对应 fusion student 模型
- **AND** 评估流程 MUST 只准备该配置启用的模态输入

#### Scenario: dry-run 使用虚拟 canonical 配置
- **WHEN** 用户运行 `python scripts/train.py --config configs/fusion/gps_mmwave_logits_kd.yaml --dry-run`
- **THEN** 系统 MUST 先生成 canonical 配置，再应用 dry-run 覆盖
- **AND** dry-run MUST 使用 synthetic dataset、单 epoch 和关闭 worker 的现有行为

#### Scenario: 保存完整 final config
- **WHEN** 使用虚拟 canonical 配置完成训练
- **THEN** 系统 MUST 在运行目录保存完整解析后的 `final_config.yaml`
- **AND** `final_config.yaml` MUST 包含训练复现所需的全部字段，而不是只保存虚拟路径或生成规则

#### Scenario: CLI override 覆盖虚拟配置
- **WHEN** 用户通过 `--override` 或点式未知参数覆盖虚拟 canonical 配置字段
- **THEN** 系统 MUST 在生成 canonical 配置之后应用这些覆盖
- **AND** 覆盖优先级 MUST 与实体 YAML 配置保持一致
