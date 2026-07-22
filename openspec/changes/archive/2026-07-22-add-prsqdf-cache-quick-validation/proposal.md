## Why

现有 PGCD 快筛在 backbone 内联合训练质量 Router，不能隔离判断 pre-prototype 传感器状态、风险监督类型和样本级动态降权各自的价值。需要冻结已完成的 C0 corruption-augmentation + global-prior checkpoint，以一次共享缓存支持 Q0--Q5 的单 seed inner 快速验证，并用 Dynamic 对 Global Mean/Prior Only 的替换实验直接否证或支持动态质量机制。

## What Changes

- 新增 claim-ineligible 的 PR-SQDF 缓存协议：只读取 MMW 的 image、radar、gps、lidar、64 类 beam index 和既有 beam topology，显式拒绝 channel、CSI、path、gain 和 beam-power 字段。
- 从冻结 C0 checkpoint 对确定性 clean/corrupted views 分片提取 pooled pre-prototype feature、prototype feature、block logits、global prior、availability、传感器统计和 train-only 风险 target；训练 Q1--Q5 时不再运行 backbone。
- 增加共享轻量 quality head、非负有上界的逐模态 beta，以及只能在 global block prior 上降权的 masked fusion；Q0 为无训练的 prior-only 锚点。
- 增加 Q1--Q5 固定输入/target 映射、一次 train-only loss 量级校准、validation-best 选择、D0--D3 替换、E0--E6 评测、质量/梯度/效率诊断和预注册 success gates。
- 对首次 batch-2048 快筛暴露的更新预算不足进行纠正：复用同一冻结缓存，以 batch 256 和最低 10 epoch 独立重跑，并显式记录有效样本数与 optimizer step 数。
- 增加 GPU0--5 预处理与六方向 launcher；单任务失败不终止其他任务，运行结束后停止，不自动启动 multi-seed、outer test 或下一轮训练。

## Capabilities

### New Capabilities

- `prsqdf-cache-quick-validation`: 定义冻结 C0 的共享 block 缓存、风险监督质量头、有界 prior correction、固定 Q0--Q5 快筛和统一诊断协议。

### Modified Capabilities

- `t2-baseline-surface`: 将 PR-SQDF 作为 active T2 inner/development 研究任务纳入可追溯 current surface，同时保持四模态和 canonical recipe 边界。
- `training-evaluation-runtime`: 允许冻结 C0 后仅从审计过的本地缓存训练小型质量头，并规定 validation-best、动态替换与 claim-ineligible 停止边界。

## Impact

变更涉及 `src/kd_sensing/models/` 的独立 PR-SQDF quality/fusion 组件、`analysis/` 的缓存和轻量训练评测程序、`scripts/` 的 GPU0--5 launcher、对应测试和 OpenSpec。实现复用现有 C0 checkpoint、MMW loader、PGCD 退化生成器与 topology helper，不新增第三方依赖、不新增 public CLI、不修改 canonical recipe，也不提交 `dataset/`、`outputs/`、缓存、日志或 checkpoint。
