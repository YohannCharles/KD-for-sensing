## MODIFIED Requirements

### Requirement: 三方法多随机种子公平训练
系统 MUST 支持在同一 MMW 15-domain、四传感器、domain-balanced sampler、缺失增强、40 epoch 和固定 `last.pth` 协议下运行 T2、AMBER-Full 与 RMBP-MM seeds1-3；S1 MUST 作为同架构无 temporal-superset consistency 对照参与需要四法比较的 matrix。每个方法 MUST 从 tracked canonical recipe 或 tracked shared base 解析，seed MUST 控制模型训练、domain sampler 和 temporal missing 随机性，但 MUST NOT 改变数据 split 或样本 inventory。

#### Scenario: launcher 不读取历史 resolved config
- **WHEN** launcher 为 T2、S1、AMBER-Full 或 RMBP-MM 生成 seed 配置
- **THEN** 生成输入 MUST 全部来自 tracked `configs/mmw/` 或 `configs/fusion/amber_full_architecture.yaml`
- **AND** launcher MUST 不读取 `outputs/` 下的 YAML、checkpoint 或 final config

#### Scenario: seed 不改变验证样本
- **WHEN** launcher 分别生成同一方法的 seed1、seed2 和 seed3 配置
- **THEN** 三者 MUST 具有相同 15-domain inventory、split 路径和固定数据 seed
- **AND** experiment、domain sampler 和 temporal missing 的行为 seed MUST 分别为 1、2 和 3
