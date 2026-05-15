## Why

参考 arXiv:2603.25799 的 occlusion-aware multimodal beam prediction 思路，当前项目只把 DeepSense6G 样本监督为未来 beam 分类，尚不能显式学习遮挡状态和二维位置。为 Scenario 31 的五模态 fusion 增加遮挡检测与位置估算头，可以把 mmWave sweep、GNSS/GPS 和感知模态的几何信息纳入同一个多任务目标，用于提升遮挡鲁棒性并产出更可解释的定位诊断。

## What Changes

- 在 DeepSense6G dataset 中提供可选的遮挡标签与二维位置目标：
  - 遮挡标签由当前/目标 mmWave 64-beam power vector 的最大接收功率自动派生，阈值默认使用训练 split 的固定分位数。
  - 位置目标优先复用现有 GPS-Rel-Polar/GNSS 本地坐标能力，按配置返回与预测 horizon 对齐的 `[x, y]` 或等价本地平面坐标。
- 在 CLS-token Transformer fusion 模型上增加可选 auxiliary heads：
  - `occlusion_head` 输出每个预测时隙的 blockage/occlusion logit。
  - `position_head` 输出每个预测时隙的二维位置估计。
  - 默认关闭时，beam logits 输出契约不变。
- 训练流程支持 beam CE + 遮挡 BCE + 位置 MSE 的多任务损失权重配置，并在日志中记录各分量。
- 验证/评估流程输出遮挡 accuracy/F1、位置 RMSE，并保留既有 Top-K、DBA 和 checkpoint 行为。
- 增加配置、测试和文档，使五模态 fusion 可复现实验能启用该多任务路线。

## Capabilities

### New Capabilities
- `multi-task-occlusion-position-learning`: 定义遮挡标签、位置目标、辅助头、多任务损失、评估指标和配置契约。

### Modified Capabilities
- `modality-aware-data-loading`: dataset 需要在启用多任务监督时返回遮挡和位置目标，并保持未启用时的按模态懒加载行为。
- `cls-token-transformer-fusion`: CLS-token Transformer fusion 需要支持可选遮挡检测头和位置估算头，同时保持既有 beam logits 契约。
- `configurable-multimodal-fusion`: fusion 配置需要暴露多任务监督、loss 权重和 recommended 五模态配置入口。
- `experiment-workflow`: 训练、验证、评估和日志需要支持多任务 loss 与辅助指标。

## Impact

- Affected code:
  - `src/kd_sensing/data/datasets/deepsense6g.py`
  - `src/kd_sensing/data/samples.py`
  - `src/kd_sensing/data/transform_ops/mmwave.py`
  - `src/kd_sensing/engine/batch.py`
  - `src/kd_sensing/engine/runtime.py`
  - `src/kd_sensing/engine/trainer.py`
  - `src/kd_sensing/engine/validator.py`
  - `src/kd_sensing/evaluation/metrics.py`
  - `src/kd_sensing/models/fusion/cls_token_transformer.py`
  - `src/kd_sensing/engine/model_output.py`
  - canonical config helpers and fusion YAML/virtual config paths.
- Tests should cover label generation thresholds, missing-column failure modes, model output shapes, loss weighting, evaluation metrics, and default beam-only backward compatibility.
- No breaking change is intended. Existing configs without the new multi-task flags must continue to train and evaluate as beam-only experiments.
