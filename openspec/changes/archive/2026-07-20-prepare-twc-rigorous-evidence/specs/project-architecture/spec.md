## MODIFIED Requirements

### Requirement: 包导入图收敛到 T2/baseline 与双数据集闭包
`kd_sensing` MUST 只增加 MaskTrain availability-normalized fusion core 和 AMR Gaussian uncertainty fusion core 及必要 auxiliary loss。二者 MUST 通过 current registry 和 `modular_sequence` staged forward 接入；系统 MUST 不恢复 retired AMR whole-model、旧 runner、BEV-Fusion 或额外 public CLI。

#### Scenario: 导入扩展 baseline surface
- **WHEN** 默认组件被导入
- **THEN** 两个新 representation core MUST 可由 registry 构建
- **AND** 未选择它们的现有 T2/AMBER/RMBP path MUST 不需要新增输入或输出字段
