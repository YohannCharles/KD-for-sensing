## Why

当前 `fair_gps_biased` Stage 2 复用 JEPA `context_encoder` 时，将每帧 196 个视觉 patch token 直接 mean pool 为 `[B,T,64]`。beam index 预测高度依赖 UE/车辆在图像中的局部位置、遮挡、道路结构、朝向以及与 BS 的相对方位，简单平均会抹平“与当前几何状态最相关的 patch 在哪里”，限制 GPS-biased JEPA 下游收益。

本 change 在不重训 JEPA Stage 1 的前提下，为 `fair_gps_biased` 下游加入 GPS-query Attention Pooling：用 GPS/运动表征生成 query，从 JEPA patch tokens 中自适应读取与当前 beam 几何相关的局部视觉信息。

## What Changes

- 新增可注册、可测试的 `GPSQueryPool` 模块，输入 JEPA patch tokens `[B,T,N,D]` 与 GPS/motion 特征 `[B,T,D]`，输出默认 `[B,T,D]` image feature，并可保留 attention map 诊断。
- 扩展 `jepa_context_image` encoder：默认继续使用 `pooling: mean`；显式配置 `pooling: gps_query_attention` 时，forward 必须接收 GPS 条件特征并使用 query attention pooling 替代 mean pooling。
- 扩展 `ModularSequenceModel` 的 encoder 调用路径，使 image encoder 可在声明需要 GPS 条件时接收同 batch/time 的 GPS 条件张量，同时保持普通 encoder 的单输入调用兼容。
- 在 `configs/fusion/experiments/jepa_image_gps/` 基于 `fair_gps_biased` 新增 GPS-query pooling 主配置与必要 ablation 配置，默认复用 GPS-biased JEPA checkpoint、训练 recipe、BeamBench-fair/2604-style 数据口径和 beam label space。
- 新增 focused tests 覆盖 `GPSQueryPool` shape、K-query 聚合、缺失 GPS 条件报错、`jepa_context_image` mean 默认兼容、modular image+GPS forward smoke、配置加载与 runtime metadata。
- 不重训 JEPA Stage 1；不改变 checkpoint schema；不新增旧入口、KD/distillation、HiST/Hist、Top8 selector 或 residual correction 路线。

## Capabilities

### New Capabilities

- `gps-query-jepa-pooling`: 定义 JEPA 下游 GPS-query Attention Pooling 的模型、配置、诊断和实验矩阵契约。

### Modified Capabilities

- `gps-conditioned-jepa-pretraining`: 扩展 JEPA context encoder 下游复用契约，使 `jepa_context_image` 在默认 mean pooling 之外支持显式 GPS-query pooling，并明确该能力仍只复用 `context_encoder` 权重。
- `modular-sequence-model`: 扩展模块化序列模型的 encoder 调用契约，使声明条件输入需求的 encoder 能接收同一 batch/time 的其它模态条件特征。

## Impact

- 影响代码：`src/kd_sensing/models/jepa.py`、`src/kd_sensing/models/modular.py`、`src/kd_sensing/engine/run_metadata.py`、相关注册表导入与 focused tests。
- 影响配置：`configs/fusion/experiments/jepa_image_gps/` 下 `fair_gps_biased` 派生配置、README 与 JEPA downstream ablation 矩阵。
- 影响验证：需要运行 JEPA/GPS-query focused tests、配置加载 smoke、forward smoke，以及 OpenSpec strict validate。
- 运行产物仍写入 `outputs/` 或配置指定目录；不提交 checkpoint、cache、日志或训练输出。
