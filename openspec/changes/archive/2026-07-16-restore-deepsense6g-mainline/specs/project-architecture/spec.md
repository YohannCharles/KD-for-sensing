## MODIFIED Requirements

### Requirement: 包导入图收敛到 T2/baseline 与双数据集闭包

`kd_sensing` MUST 只保留 T2、S1、AMBER-Full、RMBP-MM 所需的数据、模型、loss、训练、评估、预处理和通用 owner。共享 owner MUST 仅无条件导入 MMW 与受限 DeepSense6G 四模态主线所需模块，且 MUST 不无条件导入 retired family。

#### Scenario: 导入 core surface

- **WHEN** 用户导入 config、registry 或训练入口
- **THEN** 导入 MUST 不读取数据、权重或输出目录
- **AND** MMW 与 DeepSense6G 的 config load 和 synthetic forward MUST 不要求 retired module
