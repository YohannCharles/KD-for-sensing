## Context

项目当前的单模态和 fusion 模型都遵循 `(pred, features, output_features)` 输出契约，并在 `features` 之后进入 LayerNorm、GRU、时序增强模块和 classifier。用户希望把除 mmWave 外的 image、radar、GPS、LiDAR 改成 M2BeamLLM 论文的 GRU 前 sensing data encoding 方式，同时明确“最好新添，而不是覆盖”。因此本设计采用新增 encoder/profile 的方式，让既有基线配置不受影响。

M2BeamLLM 的相关范围只取 sensing data encoding：image 用 ImageNet 标准化和 ResNet-18；radar 从 raw radar 经 Range/Angle FFT 得到 RA map 后 CNN 编码；LiDAR 从点云构造单通道 256x256 histogram 再用改造 ResNet-18；GPS 先 min-max 归一化再用 MLP。论文后续的 CLIP 式 alignment、multi-head fusion、LLM backbone、SFT 和 inverse normalization 不纳入本变更，因为本项目要保持 GRU 及之后部分不变。

## Goals / Non-Goals

**Goals:**

- 新增 M2BeamLLM 风格的 image、radar、GPS、LiDAR GRU 前编码器，并保持输出形状 `[B, T, feature_size]`。
- 新增单模态和 fusion 可选入口，使用户可以显式启用新 encoder，而默认配置仍使用当前实现。
- 保持 GRU、attention/classifier、KD distiller、训练循环、评估指标的既有契约不变。
- mmWave 不做 encoder 替换，也不因为 fusion 中启用其它 M2BeamLLM encoder 而改变 mmWave 分支。

**Non-Goals:**

- 不实现 M2BeamLLM 的 LLM backbone、SFT、input/output projection 或 inverse normalization。
- 不把现有 `image_teacher`、`radar_teacher`、`gps_teacher`、`lidar_teacher` 等默认注册名改为新实现。
- 不重写现有 cache 策略；只在必要时补充新 encoder 所需的轻量 artifact 或输入适配。
- 不承诺复现 M2BeamLLM 论文完整结果，因为后端预测网络仍是本项目 GRU 体系。

## Decisions

### 1. 通过新增注册名启用，而不是覆盖现有模型

新增 `m2beamllm_image_teacher`、`m2beamllm_radar_teacher`、`m2beamllm_gps_teacher`、`m2beamllm_lidar_teacher` 以及对应 student 或共享 encoder profile。fusion 侧新增 `encoder_profile: m2beamllm` 或独立 `m2beamllm_fusion_teacher` / `m2beamllm_fusion_student`，具体实现时优先选择和现有构建器改动最小的方案。

理由：当前配置和测试已经依赖既有注册名的行为。新增入口能满足对照实验和回滚需求。

备选方案是直接替换现有 feature extractor。该方案改动少，但会破坏现有 baseline，不符合用户“新添”的要求。

### 2. 把“GRU 前编码器”做成可复用模块

新增 `src/kd_sensing/models/m2beamllm_encoders.py` 或等价模块，提供四个 encoder 类：

- `M2BeamLLMImageEncoder(feature_size, image_channels=3, pretrained=True)`
- `M2BeamLLMRadarEncoder(feature_size, radar_input_mode=...)`
- `M2BeamLLMLidarEncoder(feature_size, lidar_channels=1)`
- `M2BeamLLMGpsEncoder(feature_size, gps_input_size=2, hidden_dims=(32, 64, 64))`

单模态模型和 fusion 模型复用这些 encoder，避免在多个文件中复制论文结构。

### 3. ResNet-18 依赖显式化

image 和 LiDAR encoder 需要 ResNet-18。实现时应增加 `torchvision` 依赖，并在构建预训练权重时兼容新旧 torchvision API。如果运行环境无法下载 pretrained 权重，配置必须允许 `pretrained: false` 或使用本地缓存，避免训练入口因为网络不可用而失败。

### 4. 雷达输入契约分两层处理

M2BeamLLM 论文从 raw radar 做 Range FFT、DC removal、Angle FFT，形成 RA map 后编码。当前项目的 dataset/batch 已经常用 `radar_ra` 和 `radar_da` 拼接后的 map 输入。实现应支持：

- `radar_input_mode: ra_map`：复用现有预处理 RA map，作为默认最小可运行路径。
- `radar_input_mode: raw_fft`：在有 raw radar tensor 字段时执行 Range/Angle FFT，并只输出 RA map 编码结果。

若配置选择 `raw_fft` 但 batch 没有 raw radar 字段，系统必须给出清晰错误，而不是静默退化。

### 5. LiDAR histogram 与现有 BEV 路径兼容

论文使用点云到 1x256x256 histogram，单元点数裁剪到 5 后除以 5。当前项目已有 LiDAR BEV 懒加载和 cache。实现时优先在 dataset/preprocessing 增加可选 `lidar_encoding: m2beamllm_histogram`，输出 `[B, T, 1, 256, 256]`；如果用户继续给现有 3 通道 BEV，M2BeamLLM LiDAR encoder 必须要求显式 `lidar_channels` 匹配，避免混淆“论文 histogram”和“现有 BEV”。

### 6. GPS min-max scaler 使用训练集 artifact

M2BeamLLM GPS 使用经数据集范围统计的 min-max normalization。实现应新增或扩展 GPS scaler artifact，只在 train split fit，并在 test/eval 复用。默认 M2BeamLLM GPS 输入为经纬度 `[lon, lat]` 或 dataset 提供的等价二维坐标；现有 GPS-Rel-Polar `[B, T, 3]` 路径保持给旧模型使用。

### 7. GRU 后结构保持可验证

新增模型的 GRU、LayerNorm、attention/classifier 和 `forward` 返回值应尽量复用现有类或小型 mixin/helper。测试需要断言新旧模型的 GRU 参数和后半段模块类型不因 encoder profile 改变而变化。

## Risks / Trade-offs

- ResNet-18 增加训练显存和依赖体积 -> 提供 `pretrained: false`、冻结 backbone 或 student 轻量配置，并用小 batch smoke test 验证。
- raw radar 字段可能当前 dataset 不直接暴露 -> 先支持现有 RA map 适配，同时把 raw FFT 作为显式可选路径和后续任务。
- LiDAR 论文 histogram 与现有 3 通道 BEV 不同 -> 用独立配置字段区分，避免新 encoder 隐式吃旧 BEV。
- GPS min-max 与现有 relative_polar 标准化语义不同 -> 新增 artifact 名称和配置 profile，测试 train/test 不重新 fit。
- 论文完整框架包含 alignment/fusion/LLM，单独替换 encoder 不能等价复现论文 -> 文档中明确这是“GRU 前 encoding 对照”，不是 M2BeamLLM 完整复现。

## Migration Plan

1. 新增 encoder 模块和模型注册名。
2. 新增或扩展 dataset/preprocessing 配置字段，提供 image 标准化、LiDAR histogram、GPS min-max、radar RA/raw FFT 适配。
3. 新增单模态与 fusion 示例配置，默认 canonical 配置不变。
4. 新增单元测试和配置构建测试。
5. 更新 README 或扩展文档，说明如何启用 M2BeamLLM encoder profile，以及 mmWave 不受影响。

回滚方式：删除新增配置或改回原注册名即可；现有配置没有迁移要求。

## Open Questions

- 当前数据集是否能稳定提供 raw radar cube；如果不能，首版实现应只保证 `ra_map` 路径并把 `raw_fft` 标为需要 raw 字段。
- GPS 的二维原始坐标字段在不同 scene/split 中是否统一命名；实现前需要确认 dataset 当前返回 `gps` 的来源是否保留原始 lon/lat。
