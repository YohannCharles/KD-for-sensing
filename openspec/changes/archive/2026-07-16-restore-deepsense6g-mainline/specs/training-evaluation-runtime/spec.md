## MODIFIED Requirements

### Requirement: 评估使用 recipe 声明的数据集 protocol

evaluation MUST 复用共享四模态 batch/input contract，并以 recipe 声明的 MMW 或 DeepSense6G 数据集、checkpoint、split 和 mask identity 产出指标。DeepSense6G evaluation MUST 不执行 retired branch 或外部 teacher path。

#### Scenario: 评估 current checkpoint

- **WHEN** 用户评估任一 current recipe 的 checkpoint
- **THEN** runtime MUST 不执行 retired branch 或外部 teacher path
- **AND** 输出 MUST 带有足以比较的 recipe、dataset、scene 或 domain、seed、split 与 mask provenance
