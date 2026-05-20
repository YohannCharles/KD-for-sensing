## Why

当前项目已经支持 MMW 的 CSI 模态、RMS 归一化、pilot-based estimation noise 和 `pilot_dual_view_csi` 编码器，但数据侧仍主要把 ray tracing 生成的高质量 channel/path-level CSI 当作 clean 输入使用。为了支撑“CSI 信息上限高但在真实估计误差、多径缺失、阵列误差和同步误差下更难优化”的实验叙事，需要把 `CSI模态加噪方案.md` 中的退化策略落到项目现有 MMW/CSI 数据管线中。

## What Changes

- 新增可配置的 CSI channel degradation 能力，默认关闭，开启后在 dataset/loader 侧把 clean CSI 历史输入退化为可复现的 degraded CSI。
- 支持适合 MMW path-level channel 的退化算子：complex gain AWGN、弱路径优先 path dropout、dominant path attenuation、delay noise/quantization、AoA/AoD angle noise、antenna phase calibration error 和 CSI temporal shift。
- 提供 `clean`、`medium`、`hard` 三类质量 profile，并允许用户在 YAML 中覆盖单项参数；推荐主实验使用 medium degradation。
- 保持训练标签来自 clean future beam，不用 noisy/degraded CSI 重新生成标签；CSI RMS 统计继续基于训练 split 的 clean CSI。
- 在运行 metadata、样本 metadata 或 diagnostics 中记录 degradation profile、seed、有效参数和 temporal shift 策略，保证实验可复现。
- 增加 CSI-only 与 fusion 示例配置，覆盖 GNSS/mmWave 与 degraded CSI 的联合训练场景。
- 增加单元测试，覆盖退化算子、确定性、clean 默认行为、RMS clean 统计、temporal shift 边界和配置加载。

## Capabilities

### New Capabilities

- `csi-channel-degradation`: 定义 CSI 退化配置、退化算子、profile、随机性、metadata 记录和训练/评估行为。

### Modified Capabilities

- `csi-channel-data`: 当前要求 `csi` 样本字段始终表示 clean CSI；需要放宽为默认 clean，但在显式启用 degradation 时可返回 degraded CSI，同时必须保留 clean RMS 统计与 clean future label 契约。

## Impact

- 受影响代码：
  - `src/kd_sensing/data/transform_ops/csi.py`
  - `src/kd_sensing/data/datasets/deepsense6g.py`
  - `src/kd_sensing/data/datasets/mmw.py`
  - `src/kd_sensing/data/samples.py`
  - `src/kd_sensing/engine/data_factory.py`
  - `src/kd_sensing/engine/run_metadata.py`
  - `configs/csi/*.yaml`
  - `configs/fusion/*.yaml`
  - `tests/test_csi_modality.py`
- 不新增外部依赖；实现应使用 NumPy/PyTorch 现有能力。
- 不改变默认训练行为；未配置 degradation 时，现有 clean CSI 配置和测试应保持兼容。
- 不改变 beam label 生成规则；MMW preparation 生成的 beam power、future label 和 split CSV 继续作为监督信号来源。
