# training-evaluation-runtime Specification

## Purpose

定义 MMW T2/baseline 的共享训练与评估边界，保证 T2 的 same-model consistency 只在训练期执行，而四方法比较始终使用固定的 MMW provenance。

## Requirements

### Requirement: T2 runtime 仅保留 same-model consistency

训练 runtime MUST 只保留 T2 所需的 embedded full-modal teacher CE 和 same-model temporal superset consistency。evaluation MUST 不执行第二个 teacher forward；T2 disabled path 与 S1 MUST 不读取外部 artifact。

#### Scenario: S1 关闭 superset consistency

- **WHEN** S1 recipe 关闭 superset consistency
- **THEN** trainer MUST 不保存 superset payload 或执行第二次 model forward
- **AND** 训练仍能计算共享 beam、embedded teacher CE、BPA 和 router loss

### Requirement: 评估使用固定 MMW protocol

evaluation MUST 复用共享 MMW batch/input contract，并以 recipe 声明的固定 checkpoint、split 和 mask identity 产出指标。

#### Scenario: 评估 T2/baseline checkpoint

- **WHEN** 用户评估任一四方法 checkpoint
- **THEN** runtime MUST 不执行 retired branch 或外部 teacher path
- **AND** 输出 MUST 带有足以比较的 recipe、seed、split 与 mask provenance
