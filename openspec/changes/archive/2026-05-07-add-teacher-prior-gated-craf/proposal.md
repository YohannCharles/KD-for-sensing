## Why

当前 all-modal CRAF 在 Scenario 32 上已经能运行反事实 reliability gate，但实验显示它仍会从 noisy train loss 中高估 image、LiDAR、radar 等弱模态贡献，导致训练准确率高而验证泛化弱。方案 3 的目标是把 CRAF 改为先用单模态 teacher 验证表现建立可靠性先验，再在该先验附近学习 sample-wise residual gate，并只选择性微调强模态 encoder。

## What Changes

- 新增三阶段 CRAF 训练框架：
  - Stage 1 训练 image、radar、gps、lidar、mmwave 单模态 teacher，并保存标准化 metrics 与 checkpoint。
  - Stage 2 加载 teacher encoder 初始化 all-modal CRAF，冻结各模态 encoder，使用 teacher registry 中的 prior 初始化 gate，只训练 fusion transformer、prior residual gate 和 prediction head。
  - Stage 3 从 Stage 2 best checkpoint 选择性解冻 GPS/mmWave encoder，小学习率微调，image/LiDAR/radar 默认继续冻结。
- 新增 teacher registry 构建流程，从每个 teacher 的 `metrics.json` 和 checkpoint 生成 `outputs/scene32/teacher_registry.json`，支持手动 prior 和基于验证指标的 metric prior。
- 新增 `PriorResidualGate`、prior regularization loss、teacher encoder loader 和可选 reliability-weighted KD loss。
- 扩展 CRAF gate 类型，支持 `none`、`fixed_prior`、`prior_residual_sigmoid` 和旧 gate，用于主实验与消融。
- 默认 Stage 2/3 不启用 counterfactual 和 KD；新增 relative/shuffle counterfactual 与 reliability-weighted KD 作为后续显式 ablation，不阻塞主流程。
- 新增 Scene32 teacher、Stage 2、Stage 3 和 teacher/prior 消融配置，并保留已有 token transformer 与 fixed prior sanity baseline。
- 扩展训练日志和验证诊断，记录 gate/prior/residual、teacher load/freeze 状态、分项 loss，以及 strong-only、weak-only、GPS、mmWave 等模态组合评估结果。

## Capabilities

### New Capabilities

- `teacher-prior-gated-craf`: 描述 teacher 初始化、teacher registry、prior residual gate、prior regularization、选择性微调、可选 KD/counterfactual 和诊断日志的行为契约。

### Modified Capabilities

- `configurable-multimodal-fusion`: CRAF fusion 配置需要支持 teacher-init/prior gate/选择性微调配置、gate 类型选择、Stage 2/3 主实验和必要消融入口。
- `experiment-workflow`: 训练与评估流程需要支持 teacher registry 构建、Stage 2 encoder 加载冻结、Stage 3 参数组和选择性解冻、分项日志及模态组合验证。
- `component-registry`: 新增模型、gate、loss、teacher loader、KD loss 和 counterfactual helper 必须通过现有注册/默认导入边界接入，保持模块可测试。
- `experiment-artifact-registry`: 训练产物需要支持从单模态 teacher metrics/checkpoint 汇总 teacher reliability registry，并保留 checkpoint metadata 与场景隔离语义。

## Impact

- 影响代码：`src/kd_sensing/models/`、`src/kd_sensing/models/fusion/`、`src/kd_sensing/distillation/`、`src/kd_sensing/engine/`、配置解析、optimizer 构建、训练日志聚合和评估路径。
- 影响脚本：新增或扩展 `scripts/build_teacher_registry.py`、可选 teacher logits 导出和运行汇总脚本；训练入口继续优先使用 `scripts/train.py` 或 `python -m kd_sensing.cli.train`。
- 影响配置：新增 Scene32 单模态 teacher、Stage 2 teacher-init prior、Stage 3 selective finetune、teacher-init/fixed-prior/no-prior 消融配置。
- 影响输出：新增 `teacher_registry.json`、teacher load/freeze 诊断、gate/prior/residual 诊断、分项 loss 和模态组合验证指标。
- 兼容性：legacy 单模态、fusion、KD、token transformer、已有 CRAF 稳定化配置和 fixed-prior sanity 配置不得被默认行为改变。
- 依赖：不新增外部运行时依赖；继续使用 PyTorch、现有配置系统和现有 OpenSpec/实验目录约定。
