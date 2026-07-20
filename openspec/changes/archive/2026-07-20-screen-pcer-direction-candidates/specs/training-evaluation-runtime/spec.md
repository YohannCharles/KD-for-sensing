## MODIFIED Requirements

### Requirement: T2 runtime 仅保留 same-model consistency

训练 runtime MUST 仅保留 T2 所需的 embedded full-modal teacher CE、same-model temporal superset consistency，以及 active PCER direction search 的同模型 full-to-masked、on-policy cached-evidence target 和 balanced LOMO consistency。Evaluation MUST 不执行训练 target forward；T2 disabled path 与 S1 MUST 不产生额外 forward 或外部 artifact 读取。

#### Scenario: S1 关闭 superset consistency

- **WHEN** S1 recipe 关闭 superset consistency 且不属于 direction search
- **THEN** trainer MUST 不保存 superset payload 或执行第二次 model forward
- **AND** 训练仍能计算共享 beam、embedded teacher CE、BPA 和 router loss

#### Scenario: Direction search target

- **WHEN** B2/B3/B4 计算 supervised route target
- **THEN** target MUST detach 且 predicted router MUST 保持梯度
- **AND** on-policy removal MUST 只重跑缓存 evidence 上的轻量 router/fusion，不得重复 backbone

### Requirement: 评估使用 recipe 声明的数据集 protocol

evaluation MUST 复用共享四模态 batch/input contract，并以 recipe 声明的 MMW 或 DeepSense6G 数据集、checkpoint、split 和 mask identity 产出指标。PCER direction search MUST 使用与历史 quick PCER 相同的 development split 和 deterministic S0-S5 mask，并保留每个 S3 缺失模态。DeepSense6G evaluation MUST 不执行 retired branch 或外部 teacher path。

#### Scenario: 评估 current checkpoint

- **WHEN** 用户评估任一 current recipe 或 active direction-search checkpoint
- **THEN** runtime MUST 不执行 retired branch 或外部 teacher path
- **AND** 输出 MUST 带有足以比较的 recipe、dataset、scene 或 domain、seed、split、mask 与 checkpoint provenance
