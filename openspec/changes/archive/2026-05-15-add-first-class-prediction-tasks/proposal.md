## Why

当前 `occlusion` 和 `position` 只能作为 beam 训练路径上的 auxiliary heads 运行，单任务实验需要用 `loss.alpha: 0.0` 间接关闭 beam loss，语义不清，也无法把 loss、metrics、early stopping 和配置矩阵作为一等实验对象。为了研究多任务模态失衡，需要把 `beam`、`occlusion`、`position` 和 `multitask` 明确建模为一等预测任务，同时保留现有 `fusion` 输入路由。

## What Changes

- 增加一等预测任务抽象，区分“输入路由/模态任务”和“预测目标任务”：
  - `experiment.task` 继续表达现有输入路由，例如 `image`、`radar`、`gps`、`lidar`、`mmwave`、`fusion`。
  - 新增 `experiment.objective` 表达预测目标，例如 `beam`、`occlusion`、`position`、`multitask`。
- 新增 task/objective registry 或等价调度层，用于统一管理 target preparation、model output selection、loss、metrics、validation aggregation 和 early stopping metric aliases。
- 将 `occlusion` 和 `position` 从 auxiliary-only loss 提升为 primary objective：
  - `occlusion` 使用遮挡 logits、BCE/pos-weight、accuracy、blocked-class F1，并支持 AUROC/AUPRC 预留扩展。
  - `position` 使用二维位置输出、MSE/SmoothL1、RMSE/MAE，并按 horizon 输出指标。
  - `multitask` 使用 beam、occlusion、position 的加权组合，并分别报告各任务指标。
- 保留现有 auxiliary-head 兼容配置，但推荐新实验使用 `experiment.objective`，不再用 `loss.alpha: 0.0` 表达单任务 occlusion/position。
- 扩展 canonical/virtual YAML 配置，提供五模态和关键模态子集的 `beam`、`occlusion`、`position`、`multitask` 入口，支撑多任务模态失衡实验矩阵。
- 扩展训练、验证、评估和产物记录，使每个 objective 都有清晰的主 loss、主 metric、early stopping、TensorBoard scalar、`training_outputs.npz` 字段和 final config metadata。

## Capabilities

### New Capabilities

- `first-class-prediction-tasks`: 定义 `experiment.objective`、预测任务注册/调度、任务专属 targets、loss、metrics、early stopping 和多任务组合契约。

### Modified Capabilities

- `experiment-workflow`: 训练、验证、评估和日志流程需要按 `experiment.objective` 选择主目标、主 loss、主指标和 early stopping。
- `configurable-multimodal-fusion`: fusion canonical/virtual 配置需要暴露 prediction objective 矩阵，并支持多任务模态失衡实验命名。
- `modality-aware-data-loading`: dataset 和 batch 准备需要按 objective 返回 beam、occlusion、position 或 multitask targets，并保持未启用目标不读取额外标签。
- `cls-token-transformer-fusion`: CLS-token fusion 的 `occlusion` 和 `position` heads 需要能作为 primary objective 输出使用，同时保持 beam-only 和 auxiliary 兼容。

## Impact

- Affected code:
  - `src/kd_sensing/config/io.py`
  - `src/kd_sensing/config/canonical.py`
  - `src/kd_sensing/engine/batch.py`
  - `src/kd_sensing/engine/runtime.py`
  - `src/kd_sensing/engine/trainer.py`
  - `src/kd_sensing/engine/validator.py`
  - `src/kd_sensing/engine/evaluator.py`
  - `src/kd_sensing/engine/auxiliary.py`
  - `src/kd_sensing/engine/optim.py`
  - `src/kd_sensing/evaluation/metrics.py`
  - `src/kd_sensing/models/fusion/cls_token_transformer.py`
  - `src/kd_sensing/data/datasets/deepsense6g.py`
  - canonical fusion YAML/virtual config paths and README/docs.
- Tests should cover config validation, objective-specific loss/metric selection, single-task occlusion/position smoke training, multitask training, evaluator artifact reuse, and backward compatibility for existing `experiment.task: fusion` beam-only configs.
- No breaking change is intended for existing configs. If `experiment.objective` is omitted, the system MUST default to `beam`.
