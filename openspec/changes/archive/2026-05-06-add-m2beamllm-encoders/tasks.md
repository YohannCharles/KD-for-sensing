## 1. 依赖与入口设计

- [x] 1.1 确认 `conda run -n kd_mm_beam python -c "import torch, torchvision"` 可用；若不可用，将 `torchvision` 加入 `pyproject.toml` 并在环境中安装验证
- [x] 1.2 确定最终启用方式：新增 `m2beamllm_*` 模型注册名，或在 fusion 中新增 `encoder_profile: m2beamllm`
- [x] 1.3 梳理现有 image/radar/GPS/LiDAR/mmWave 配置，确认默认配置不需要迁移

## 2. M2BeamLLM Encoder 模块

- [x] 2.1 新增 `src/kd_sensing/models/m2beamllm_encoders.py`，实现共享 shape 校验、序列 flatten/restore 和 ResNet-18 构建工具
- [x] 2.2 实现 `M2BeamLLMImageEncoder`：RGB 适配、224x224 resize、ImageNet normalize、ResNet-18 去分类头、FC/ReLU 投影到 `feature_size`
- [x] 2.3 实现 `M2BeamLLMRadarEncoder` 的 `ra_map` 路径：接收 `[B, T, C, H, W]` 雷达 map，经 CNN/pooling/MLP 输出 `[B, T, feature_size]`
- [x] 2.4 为 `M2BeamLLMRadarEncoder` 增加 `raw_fft` 显式路径的接口和缺失 raw 输入错误；若 raw tensor 字段可用，再实现 Range/Angle FFT 到 RA map
- [x] 2.5 实现 `M2BeamLLMLidarEncoder`：接收单通道 256x256 histogram，改造 ResNet-18 输入层并投影到 `feature_size`
- [x] 2.6 实现 `M2BeamLLMGpsEncoder`：二维 GPS min-max 归一化后的 MLP、LayerNorm、GELU 投影到 `feature_size`

## 3. 数据与预处理适配

- [x] 3.1 为 M2BeamLLM image 路径确认 dataset 返回原图或 motion mask 的通道语义，并通过配置显式控制单通道到 RGB 的适配
- [x] 3.2 新增 LiDAR `m2beamllm_histogram` 预处理/读取选项，生成 `[T, 1, 256, 256]` histogram，点计数裁剪到 5 后除以 5
- [x] 3.3 新增或扩展 GPS min-max scaler artifact：只在 train split fit，并在 test/eval dataloader 复用
- [x] 3.4 明确 radar `ra_map` 与 `raw_fft` 的 batch 字段来源；首版至少保证 `ra_map` 路径可运行，`raw_fft` 缺输入时报清晰错误
- [x] 3.5 确认 mmWave dataset、scaler 和 batch 准备路径不因 M2BeamLLM profile 发生变化

## 4. 单模态模型注册

- [x] 4.1 新增 image M2BeamLLM teacher/student 模型或可复用基类，复用现有 GRU、attention/classifier 和输出契约
- [x] 4.2 新增 radar M2BeamLLM teacher/student 模型，保持 radar-only batch、GRU 后结构和 KD 输出契约兼容
- [x] 4.3 新增 GPS M2BeamLLM teacher/student 模型，接收 `[B, T, 2]` GPS 坐标并保持 GRU 后结构不变
- [x] 4.4 新增 LiDAR M2BeamLLM teacher/student 模型，接收 M2BeamLLM histogram 输入并保持 GRU 后结构不变
- [x] 4.5 在 `src/kd_sensing/models/__init__.py` 和注册表导入路径中暴露新增模型，确保 `MODELS.build` 可按名称构建

## 5. Fusion 集成

- [x] 5.1 在 fusion teacher 中支持 M2BeamLLM encoder profile，使 image/radar/GPS/LiDAR 分支可选择新 encoder
- [x] 5.2 在 fusion student 中支持 M2BeamLLM encoder profile 或新增对应注册名，保持 lightweight student 与输出契约
- [x] 5.3 确保 fusion 中包含 mmWave 时仅非 mmWave 分支切换到 M2BeamLLM encoder，mmWave 分支仍使用现有 `MmWaveFeatureExtractor`
- [x] 5.4 更新 fusion 输入检查，确保启用 `raw_fft` 或 LiDAR histogram 时缺少必要字段会给出清晰错误

## 6. 配置与文档

- [x] 6.1 新增 image/radar/GPS/LiDAR 单模态 M2BeamLLM encoder 示例配置，默认 `gru_params` 保持 `[64, 64, 1]`
- [x] 6.2 新增至少一个 fusion M2BeamLLM encoder 示例配置，覆盖 `image_radar_gps_lidar` 和包含 mmWave 的排除规则
- [x] 6.3 更新 README 或 docs，说明这是 GRU 前 encoding 对照，不是完整 M2BeamLLM LLM backbone 复现
- [x] 6.4 文档中说明 raw radar、LiDAR histogram、GPS min-max artifact 和 ResNet-18 pretrained 权重的使用限制

## 7. 测试与验证

- [x] 7.1 新增 encoder shape 单元测试，并用 `conda run -n kd_mm_beam pytest tests/test_m2beamllm_encoders.py` 验证
- [x] 7.2 新增模型注册和 forward contract 测试，覆盖四个非 mmWave 单模态 M2BeamLLM 模型
- [x] 7.3 新增 fusion profile 测试，断言 mmWave 分支不被 M2BeamLLM encoder 替换
- [x] 7.4 新增 GPS min-max train/test 复用测试，断言 test split 不重新 fit
- [x] 7.5 新增默认配置回归测试，断言未显式启用 M2BeamLLM encoder 时现有配置仍构建旧模型
- [x] 7.6 运行相关测试集：`conda run -n kd_mm_beam pytest tests/test_student_configs.py tests/test_gps_modality.py tests/test_lidar_modality.py tests/test_mmwave_modality.py`
- [x] 7.7 运行一次小 batch smoke test 或构建测试，确认新增 M2BeamLLM 示例配置可加载并完成一次 forward
