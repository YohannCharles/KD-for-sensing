## ADDED Requirements

### Requirement: 包导入图收敛到 T2/baseline 闭包
当前 `kd_sensing` 包 MUST 只保留 T2、S1、AMBER-Full、RMBP-MM 所需的数据、模型、loss、训练、评估、预处理和通用工具 owner。共享 owner MUST 不无条件导入 CSI、physics、final-C2、历史诊断或其它已退役 family。

#### Scenario: 核心 T2 导入不加载退役 family
- **WHEN** 配置 loader、T2/baseline registry 或训练入口导入
- **THEN** 它们 MUST 不要求 retired family 的模块、配置或可选依赖存在
- **AND** 四方法的 config load 和 synthetic forward MUST 保持可用
