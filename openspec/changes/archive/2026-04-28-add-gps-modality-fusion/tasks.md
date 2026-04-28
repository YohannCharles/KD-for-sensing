## 1. 数据与预处理

- [x] 1.1 更新 `src/kd_sensing/preprocessing/sequences.py`，支持在序列 CSV 中输出 `gps1..gpsN` 和 `bs_gps1..bs_gpsN` 或等价 GPS 路径列
- [x] 1.2 更新预处理配置或新增 GPS 序列配置，使用户可通过 `scripts/preprocess.py` 生成携带 GPS 列的 train/test sequence CSV
- [x] 1.3 在 `src/kd_sensing/data/samples.py` 扩展 `SequenceSamples`，可选保存 UE/BS GPS 路径且不破坏旧 CSV 读取
- [x] 1.4 在 `src/kd_sensing/data/transforms.py` 或新 GPS 数据模块中实现 GPS 文本读取，支持普通浮点和科学计数法
- [x] 1.5 添加 UTM/XY 转换实现；若使用外部依赖 `utm`，更新项目依赖并用 `conda run -n kd_mm_beam python -c "import utm"` 验证环境可导入
- [x] 1.6 收敛 GPS 特征构造公开路径，只保留 GPS-Rel-Polar `[dist, sin_theta, cos_theta]`，并移除或拒绝 `raw`、`utm`、`relative`、`motion`、`motion_smooth`
- [x] 1.7 实现 GPS scaler，只允许 train split fit，并支持 test/val split 复用训练 scaler
- [x] 1.8 更新 `Scenario9Dataset`，在启用 GPS 时返回 `[seq_len, gps_feature_dim]` 的 `gps` float32 张量，未启用 GPS 时保持旧行为
- [x] 1.9 更新 `build_dataloaders`，确保 train dataset 的 GPS scaler 传递给 test dataset

## 2. GPS 模型

- [x] 2.1 新增 `src/kd_sensing/models/gps.py`，实现 `GpsFeatureExtractor`
- [x] 2.2 实现并注册 `GpsModalityNet`，注册名为 `gps_teacher`
- [x] 2.3 实现并注册 `GpsStudentModalityNet`，注册名为 `gps_student`
- [x] 2.4 为 GPS teacher/student 添加 `gru_params` 长度、`gru_input_size == feature_size` 和 `gps_input_size` 校验
- [x] 2.5 更新 `src/kd_sensing/models/__init__.py` 导出 GPS 模型类

## 3. 输入准备与训练流程

- [x] 3.1 在 `src/kd_sensing/engine/batch.py` 新增 `prepare_gps_inputs`，按预测窗口补齐未来 GPS zero padding
- [x] 3.2 更新 `forward_model`，支持 `task: gps` 和 fusion 多输入分发
- [x] 3.3 更新 `trainer._forward_for_task`，支持 GPS-only 和根据 fusion `modalities` 准备输入
- [x] 3.4 更新 `validator.validate`，支持 GPS-only 和可选模态 fusion 验证
- [x] 3.5 更新 `evaluator.evaluate` 所需路径，确保 GPS-only 和可选模态 fusion 可通过统一评估入口运行

## 4. 可选模态 Fusion

- [x] 4.1 在 fusion 模型中实现 `modalities` 解析和校验，拒绝空列表、重复模态和未知模态
- [x] 4.2 更新 `FusionModalityNet`，按 `modalities` 创建 image/radar/gps 分支并动态设置 fusion projection 输入维度
- [x] 4.3 更新 `StudentModalityNet`，按 `modalities` 创建轻量 image/radar/gps 分支并动态设置 fusion projection 输入维度
- [x] 4.4 保持 fusion 默认 `modalities` 为 `["image", "radar"]`，确保旧 fusion 配置无需修改即可运行
- [x] 4.5 为启用模态缺少输入的 forward 路径添加清晰错误信息

## 5. 配置与文档

- [x] 5.1 更新 `configs/gps/no_kd.yaml`，使用 `gps_teacher` 训练 GPS-Rel-Polar teacher baseline，并设置 `gps_feature_mode: relative_polar`、`gps_input_size: 3`
- [x] 5.2 更新 `configs/gps/student_no_kd.yaml`，使用 `gps_student` 训练轻量 GPS-Rel-Polar student baseline，并设置 `gps_feature_mode: relative_polar`、`gps_input_size: 3`
- [x] 5.3 更新 `configs/gps/logits_kd.yaml` 和 `configs/gps/rkd.yaml`，默认使用 GPS-Rel-Polar frozen `gps_teacher` 蒸馏 `gps_student`
- [x] 5.4 删除 raw、UTM、relative、motion、motion-smooth 的独立 GPS ablation 配置入口，仅保留 GPS-Rel-Polar 受支持配置
- [x] 5.5 更新 fusion 配置或新增示例配置，覆盖 `image+gps`、`radar+gps`、`image+radar+gps` 和默认 `image+radar`，其中所有 GPS 分支使用 GPS-Rel-Polar
- [x] 5.6 更新 README 和 `docs/extension_guide.md`，说明 GPS 预处理只保留 GPS-Rel-Polar、GPS 模型注册名和 fusion `modalities`

## 6. 测试与验证

- [x] 6.1 更新 GPS 特征构造单元测试，覆盖 GPS-Rel-Polar 输出维度、关键数值关系和非保留 `gps_feature_mode` 的拒绝行为
- [x] 6.2 新增 GPS scaler 测试，断言 test split 复用 train scaler 且不会重新 fit
- [x] 6.3 新增 GPS teacher/student 构建和 forward contract 测试
- [x] 6.4 新增 fusion `modalities` 校验测试，覆盖默认值、全部模态、双模态、单模态和非法配置
- [x] 6.5 更新旧配置兼容测试，确保 image-only、radar-only、默认 image+radar fusion 仍可构建，且 GPS 配置统一使用 GPS-Rel-Polar
- [x] 6.6 运行 `conda run -n kd_mm_beam pytest tests/test_student_configs.py` 验证既有 student 配置兼容
- [x] 6.7 运行 `conda run -n kd_mm_beam pytest` 完整验证测试套件
- [x] 6.8 运行 `openspec status --change add-gps-modality-fusion` 确认 change artifacts 已达到可实施状态
