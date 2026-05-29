## Why

当前 HiST-Beam 快速验证只证明了 hierarchical beam、shared/private 和轻量 adapter 的工程闭环，但没有显式建模论文设定中真正可迁移的角度语义、角度邻域连续性、RSU-CAV 相对几何和多模态几何一致性。现有 `v5_adapter_proto` 与 `v4_adapter` 在 quick validation 中指标几乎完全一致，说明 prototype loss 没有形成可观测贡献；同时 Multimodal-Wireless 数据已经具备 CAV/RSU LiDAR、RGB/depth camera、GPS/IMU、bbox、radar point cloud 和 V2I channel paths，应该把方法重心从“结构命名”推进到“可实现的几何迁移协议”。

## What Changes

- 重新定义 HiST-Beam transferable knowledge：coarse angular/beam semantics、angular neighborhood continuity、RSU-CAV relative geometry 和 cross-modal geometric consistency。
- 重新定义 scene-private knowledge：town/scene layout、RSU pose/local coordinate frame、local scatterer/occluder proxy，以及 coarse sector 内的 fine beam mapping。
- 新增 geometry-aware shared encoder、scene-private branch、fine mapping adapter、private prototype alignment、angular smoothing loss、multimodal geometry consistency loss 的实现契约。
- 将 prototype alignment 从当前 shared coarse prototype 的无效近似，调整为 coarse sector 条件下的 private prototype / target adapter 对齐，并要求记录 coverage、confidence、loss 生效诊断。
- 为 Multimodal-Wireless 设计数据加载与准备协议：支持已下载 sunny 单场景先生成 prepared manifest，后续 rainy/foggy/其它场景下载完成后增量加入；支持 scenario/town/weather split、LOSO split 和 target adaptation split。
- 明确论文概念与实现边界：可直接实现的内容必须来自可观测字段或派生几何量；不可直接观测的 scatterer/occluder、真实语义场景因素只能作为 proxy，不得在论文或日志中过度声明。
- 扩展 LOSO workflow，使其不再只绑定 DeepSense6G scenarios 31-34，而能以数据集 descriptor 生成 MMW scenario/town/weather 级 source/target/adapt/test 切分。
- 保留现有 DeepSense6G HiST-Beam 快速验证能力，但将其定位为历史证据和回归基线，不作为新论文主设定的唯一数据协议。

## Capabilities

### New Capabilities
- `mmw-cross-scene-adaptation-protocol`: 定义 Multimodal-Wireless 的 prepared manifest、scenario/town/weather split、target adaptation protocol、可用模态与几何派生字段契约。

### Modified Capabilities
- `hist-beam-cross-scene-adaptation`: 将 HiST-Beam 从快速验证版升级为 geometry-aware 跨场景自适应方法，增加可迁移/scene-private 知识定义、几何一致性损失、角度平滑和有效 prototype alignment 诊断。
- `cross-scene-loso-workflow`: 将 LOSO 编排从 DeepSense6G 31-34 专用扩展为可描述 MMW scenario/town/weather folds，并保留 target_test 防泄漏要求。
- `mmw-town10-dataset-preparation`: 扩展当前 Town10 skybridge 准备契约，要求显式记录 sensor/channel 场景名匹配、可增量处理下载中的条件/场景，并为跨场景适配产出稳定 manifest。

## Impact

- 影响模型与 loss：`src/kd_sensing/models/fusion/hist_beam.py`、`src/kd_sensing/engine/hist_beam_losses.py`、`src/kd_sensing/engine/hist_beam_adaptation.py`、`src/kd_sensing/engine/hist_beam_prototypes.py`。
- 影响 MMW 数据准备、dataset descriptor、dataset loader、batch profile 和配置：`src/kd_sensing/data/mmw/`、`src/kd_sensing/data/datasets/mmw.py`、`src/kd_sensing/data/layouts.py`、`configs/preprocess/mmw_town10_skybridge.yaml` 及新增 MMW adaptation 配置。
- 影响 LOSO 编排、summary 和 quick validation conclusion：现有 `kd-sensing-hist-beam-loso` 行为需要保留 DeepSense6G 回归，同时新增 MMW protocol。
- 本地数据产物仍位于 `dataset/MMW/`、`dataset/_downloads/MMW/`、`outputs/` 等 ignored 路径，不纳入源码变更。
