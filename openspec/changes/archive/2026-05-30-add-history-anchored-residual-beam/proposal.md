## Why

最近的 MMW Town10 场景重跑显示，当前 HiST-Beam/P3 以绝对 beam ID 作为迁移目标时，在 `skybridge -> crossroad` 的 source-only target test 上接近 64 类随机水平，而同场景训练和 `last_beam` 诊断基线都正常。这说明主要问题不是训练/评估代码错位，而是模型学到 source 场景的绝对 beam-ID 先验；跨场景真正可迁移的应是相对历史 beam 的传播变化、路径/几何模式和场景私有校准。

这个 change 将新增一个显式 opt-in 的 history-anchored residual beam 预测路径，把 `input_beam` 从评估诊断字段提升为可审计的历史锚定输入，并用 circular residual/delta label 替代直接绝对 beam 分类作为主迁移目标。

## What Changes

- 新增 history-anchored residual beam prediction 能力：
  - 通过配置显式启用历史 beam 输入，默认不改变现有 sensor-assisted `image`/`gps`/`lidar` 三主模态输入边界。
  - 支持 `delta = wrap(future_beam - last_beam)` 的 64 类环形 residual 目标。
  - 支持将 residual logits 重建为绝对 beam logits/prediction，用于现有 Top-1/Top-3/Top-5、NRP 和 dB loss 指标。
  - 增加 `last_beam`、Markov delta、history-absolute classifier、residual-only、residual+private calibration 等对比实验定义。
- 修改 HiST-Beam 跨场景适配要求：
  - 增加 history-anchored 模型变体和配置契约。
  - 重新约束 shared/private 解耦语义：shared branch 优先学习相对传播 residual，private branch 学 scene-specific offset/bias/temperature/prototype calibration。
  - few-shot adaptation 在 residual 模式下默认冻结 shared backbone，只训练 private adapter、residual/calibration head、logit bias、temperature 或等价低参数校准模块。
- 修改 MMW sensor-assisted 协议边界：
  - 保留现有 “last-beam diagnostic baseline 不改变模型输入语义” 要求。
  - 默认 sensor-assisted/P3 主结论不再消费 radar sensing input；radar 仍保留为显式可选的通用模态/诊断能力。
  - 新增 `history_anchored` 或等价 profile 标记；启用后该 run 不再与默认 sensor-assisted 主结论混淆，summary 必须标明它使用了历史 beam 输入。
- 修改 MMW cross-scene adaptation protocol：
  - target split 防泄漏检查必须覆盖历史 beam 窗口上下文，确保 `input_beam` 只来自样本自身历史窗口，不来自 target future/test 标签。
  - quick validation 可加入 history-anchored residual 快速矩阵，但必须与原 sensor-assisted quick validation 分开汇总。

## Capabilities

### New Capabilities
- `history-anchored-residual-beam`: 定义历史 beam 输入、环形 residual label、绝对 beam 重建、校准分支、诊断指标和最小实验矩阵。

### Modified Capabilities
- `hist-beam-cross-scene-adaptation`: 增加 history-anchored residual HiST-Beam 变体、loss、shared/private 解耦语义、few-shot private calibration 和评估产物要求。
- `mmw-sensor-assisted-beam-prediction`: 增加可审计的 history-anchored profile 边界，避免把历史 beam 输入静默归入默认 sensor-assisted 主结论。
- `mmw-cross-scene-adaptation-protocol`: 增加历史 beam 窗口防泄漏和 history-anchored quick validation 矩阵/summary metadata 要求。

## Impact

- 主要受影响代码：
  - `src/kd_sensing/models/fusion/hist_beam.py`：扩展 `HistBeamConfig` 和 `HistBeamFusionNet.forward()`，新增可选 beam-history embedding、residual head 和 calibration 输出。
  - `src/kd_sensing/engine/batch.py`：在 opt-in profile 下准备 `input_beam`/history anchor batch；默认 sensor-assisted 路径保持不变。
  - `src/kd_sensing/engine/evaluation_pass.py`：支持 residual logits 到绝对 beam 空间的重建、residual metrics、Markov/last-beam baseline 和诊断输出。
  - `src/kd_sensing/engine/hist_beam_loso_execution.py`：在 source train、source-only eval、target adaptation 和 adapted eval 中传递 history-anchor 配置与 metadata。
  - LOSO 配置和脚本：新增 history-anchored residual 快速验证配置，不覆盖现有 P3/V8 脚本语义。
- 需要新增或扩展测试：
  - circular residual label/reconstruction 单元测试。
  - batch preparation opt-in/default regression 测试。
  - model forward/loss/evaluation metrics 测试。
  - target adaptation 防泄漏和 eligibility metadata 测试。
- 不引入新的外部依赖；所有项目相关 Python 验证命令继续使用 `conda run -n kd_mm_beam`。
