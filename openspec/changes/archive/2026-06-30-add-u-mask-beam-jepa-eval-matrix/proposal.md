## Why

U-MaskBeamJEPA 已具备缺失模态训练机制，但缺少按固定缺失模式和随机缺失率系统比较的评估矩阵。该 change 用于补齐 evaluation / diagnostics / config / tests，不改变模型主干、encoder、训练 loss 或既有训练行为。

## What Changes

- 新增 U-MaskBeamJEPA missing-modality evaluation matrix，覆盖 full、single-missing、only-one-available、pair-missing 和 random missing 条件。
- 新增固定 missing pattern 生成、随机 eval mask 采样、Top-K、ECE 和 reliability-vs-error 统计工具。
- 新增可调用 runner 和包内 CLI `kd-sensing-eval-u-mask-matrix`，从 checkpoint/config 执行评估并导出 CSV、JSON 和 Markdown。
- 新增 Scenario 32 eval 配置、使用文档和 fake model 单元测试。
- 明确本 change 不包含 corruption protocol，不修正 simplified encoder，也不改变 JEPA-MSAC preprocessing 或训练目标。

## Capabilities

### New Capabilities

- `u-mask-beam-jepa-eval-matrix`: U-MaskBeamJEPA 缺失模态评估矩阵、reliability diagnostics、结果导出和非侵入式评估入口。

### Modified Capabilities

无。

## Impact

- 代码：`src/kd_sensing/eval/`、`src/kd_sensing/cli/`、`pyproject.toml`、`configs/eval/`、`tests/`、`docs/`。
- API：新增包内 CLI 和 callable eval runner；不改变已有训练入口、模型注册名、loss 配置或 checkpoint 格式。
- 数据与产物：默认评估输出写入 ignored `outputs/eval/...`；单元测试不依赖真实 DeepSense6G 数据或 checkpoint。
- 依赖：不新增第三方依赖，只使用 PyTorch 和 Python 标准库。
