## Why

BeamBench 表明 GPS 能提供低维、可解释的位置先验，但直接学习绝对 beam 在未见场景下仍会受到场景轨迹、基站朝向和标签分布偏移影响。当前项目已经有 GPS-only 模型和 GPS window 几何 baseline；下一步需要把 GPS 预测沉淀为可跨场景泛化的粗粒度 anchor，为后续“其它模态预测残差”提供稳定、可审计的第一阶段。

## What Changes

- 新增 GPS coarse anchor 预测能力：只使用 GPS/pose 或 GPS-Rel-Polar 输入，输出 coarse group logits、coarse center beam、可选 beam score 分布、anchor confidence 和 residual anchor metadata。
- 支持两类 GPS anchor：BeamBench-style 几何校准 anchor 作为强可解释基线，以及轻量 GPS neural coarse head 作为可训练变体；二者输出统一契约，便于后续 residual fusion 复用。
- 新增跨场景 LOSO/未见场景评估 profile：默认以 DeepSense6G Scenes 31-34 或本地可用 MMW scenes 做 source/target 拆分，记录 seen/unseen 场景、calibration split、target_test 防泄漏状态和 distribution-shift 诊断。
- 将 GPS anchor 纳入 HiST-Beam 训练/评估接口：允许模型 forward 和 prediction artifact 记录 `gps_anchor`，但默认不改变现有 HiST-Beam、history-anchored、GPS-only 和 GPS window baseline 行为。
- 新增 GPS anchor 指标：coarse accuracy、anchor beam circular error、anchor confidence calibration、GPS anchor oracle-usage metadata，以及以 anchor 为中心的 residual label 预览。
- 保持范围收敛：本变更只实现第一阶段 GPS 粗预测与可消费接口，不实现 image/radar/lidar/mmwave residual head 的完整训练。

## Capabilities

### New Capabilities

- `gps-coarse-anchor-prediction`: 定义 GPS 粗粒度 anchor 的输入边界、输出契约、跨场景评估、训练/校准模式、防泄漏 metadata 和 residual 接口。

### Modified Capabilities

- `hist-beam-cross-scene-adaptation`: 允许 HiST-Beam 配置显式消费 GPS coarse anchor 作为 coarse/fine 或 residual 预测的条件输入，并要求评估产物记录 anchor 字段；默认行为不变。
- `gps-modality-model`: 扩展 GPS 模型能力，支持 GPS coarse head 或 anchor export profile，保持现有 `gps_teacher`/`gps_student` beam logits forward 契约兼容。

## Impact

- 影响代码：预计新增 `src/kd_sensing/engine/gps_coarse_anchor.py` 或 `src/kd_sensing/models/heads/gps_anchor.py`，复用 `kd_sensing.baselines.gps_window` 的几何/boresight 工具和 `HistBeamFusionNet` 的 coarse/fine label 语义。
- 影响配置：新增 GPS anchor smoke、DeepSense6G Scene 31 unseen 泛化、MMW LOSO anchor profile，以及可选 HiST-Beam anchor-conditioned profile。
- 影响训练/评估：新增显式 opt-in 的 GPS anchor builder、loss、metrics 和 prediction artifact 字段；现有默认训练、GPS window CLI 和 HiST-Beam quick validation 不被静默改变。
- 影响测试：新增 anchor 输出形状、coarse label、跨场景 split、防泄漏、anchor-conditioned forward、artifact 字段和 OpenSpec 校验测试。
- 影响产物：运行输出新增 GPS anchor metrics、calibration metadata、anchor predictions 和 residual preview；不提交日志、checkpoint、cache 或本地数据。
