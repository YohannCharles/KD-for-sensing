## 1. 引用盘点

- [x] 1.1 使用 `rg -n "m2beamllm|M2BeamLLM|encoder_profile" src tests configs README.md openspec` 盘点所有引用并区分模型、fusion、配置、测试、文档和数据预处理范围
- [x] 1.2 确认哪些数据预处理分支只服务于 M2BeamLLM encoder，哪些仍被其它正式能力或配置依赖

## 2. 模型与注册入口清理

- [x] 2.1 删除 `src/kd_sensing/models/m2beamllm_encoders.py`
- [x] 2.2 从 `src/kd_sensing/models/image.py` 移除 M2BeamLLM image import、teacher/student 注册类和专用构造参数
- [x] 2.3 从 `src/kd_sensing/models/radar.py` 移除 M2BeamLLM radar import、teacher/student 注册类和 `radar_input_mode` 专用路径
- [x] 2.4 从 `src/kd_sensing/models/gps.py` 移除 M2BeamLLM GPS import、teacher/student 注册类和专用 feature extractor
- [x] 2.5 从 `src/kd_sensing/models/lidar.py` 移除 M2BeamLLM LiDAR import、teacher/student 注册类和专用 feature extractor
- [x] 2.6 从 fusion 模型移除 `encoder_profile: m2beamllm` 分支和所有 `M2BeamLLM*Encoder` 依赖，确认标准 image/radar/GPS/LiDAR/mmWave 分支仍按默认 feature extractor 构建

## 3. 配置、数据与文档清理

- [x] 3.1 删除 `configs/m2beamllm/` 示例配置目录
- [x] 3.2 删除 README 中 M2BeamLLM Encoder 对照章节和相关运行命令
- [x] 3.3 删除或退役只服务于 M2BeamLLM encoder 的 `gps_feature_mode: m2beamllm_minmax`、`lidar_encoding: m2beamllm_histogram` 相关处理；若其它能力仍依赖，则保留通用实现并移除 M2BeamLLM 命名入口
- [x] 3.4 再次运行 `rg -n "m2beamllm|M2BeamLLM|encoder_profile" src tests configs README.md`，确认没有残留可用入口或过期文档

## 4. 测试更新

- [x] 4.1 删除 `tests/test_m2beamllm_encoders.py` 中 M2BeamLLM 正向构建、shape、fusion profile 和示例配置测试
- [x] 4.2 补充或保留默认模型构建回归测试，覆盖标准单模态和 fusion 注册名
- [x] 4.3 补充退役入口失败测试，确认 `m2beamllm_*` 注册名和 `encoder_profile: m2beamllm` 不再作为支持路径成功构建

## 5. 验证

- [x] 5.1 使用 `conda run -n kd_mm_beam pytest` 运行相关测试；如全量耗时过长，至少运行模型注册、配置加载和被改测试文件对应的子集
- [x] 5.2 使用 `conda run -n kd_mm_beam python -m compileall src/kd_sensing` 或等价检查确认删除模块后不存在 import/语法错误
- [x] 5.3 运行 `openspec status --change remove-m2beamllm-encoders` 确认变更仍处于可实施状态
