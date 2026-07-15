## Why

当前 H5/P1 temporal workflow 对重叠滑窗执行逐样本分层拆分，实测所有序列组均跨越 train、validation 和 test；通用训练器还会在缺少 validation 时把 test 用于逐轮选模。继续沿用这些协议会产生不可引用的乐观指标，并使归一化与验证 loss 的口径无法证明只依赖训练数据。

## What Changes

- **BREAKING**：重叠窗口数据必须按稳定序列组拆分，并在 split artifact 中校验 sample、输入帧和 target 帧身份两两不相交；逐样本 temporal split 不再允许用于可比较证据。
- **BREAKING**：启用 checkpoint 选择或 early stopping 时必须提供独立 validation；缺少 validation 的 fixed-epoch 运行只能显式选择 `last.pth`，final test 不得参与训练期决策。
- 将 GPS、LiDAR、mmWave、CSI、position 和 occlusion 等 normalization/statistics 统一为仅从实际 train indices 拟合、向 validation/test 只读传播的 artifact 契约。
- 验证 loss 按有效 sample/token 数加权聚合，避免最后一个小 batch 改变 checkpoint 排序。
- 将现有 H5/P1 temporal 结果标为 `not_comparable`，记录泄漏原因和重跑门禁，不写入未经验证的新数值。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `temporal-window-missing`：增加组安全拆分、帧身份不相交和旧证据失效要求。
- `training-evaluation-runtime`：禁止 test-as-validation，并统一 validation loss 的样本加权口径。
- `dataset-runtime-contracts`：要求所有数据依赖的 normalization/statistics 仅从实际训练子集拟合和复用。
- `mainline-experiment-documentation`：要求受泄漏影响的 temporal evidence 降级并满足重跑门禁后才能晋级。

## Impact

- 影响 `src/kd_sensing/engine/data_factory*.py`、trainer/checkpoint/evaluation runtime、DeepSense6G normalization artifact 和 H5/P1 launcher/evaluator/summary。
- 多数仅配置 train/test 且启用 early stopping 的现有 YAML 将 fail closed，需要增加真实 validation 或显式 fixed-epoch/no-selection 语义。
- 旧 normalization artifact、H5/P1 split manifest 和相关结果不得与修复后的运行混合比较；真实数据、checkpoint、日志和重跑产物仍保留在 ignored 本地目录。
