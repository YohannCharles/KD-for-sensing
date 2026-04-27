## 1. 模型实现

- [x] 1.1 新增 `src/kd_sensing/models/radar.py`，实现并注册 `radar_teacher` 模型。
- [x] 1.2 复用或迁移 `RadarFeatureExtractor`，确保 RadarTeacher 接收 `(B, T, 2, H, W)` 雷达输入并输出 `(pred, features, enhanced_seq_out)`。
- [x] 1.3 为 `gru_params` 与 `num_heads` 增加构建期校验，错误信息明确指出无效参数。
- [x] 1.4 更新 `src/kd_sensing/models/__init__.py` 和相关包导出，确保默认组件导入后 `MODELS` 包含 `radar_teacher`。

## 2. Radar-only 任务路径

- [x] 2.1 在 `src/kd_sensing/engine/batch.py` 新增 `prepare_radar_inputs()`，复用 fusion 雷达截断、RA/DA channel 拼接和 zero padding 语义。
- [x] 2.2 扩展 `forward_model()` 支持 `task == "radar"`，只向模型传入雷达 batch。
- [x] 2.3 扩展 `trainer.py` 的训练循环，支持 radar-only student/teacher forward 和 KD dummy teacher 路径。
- [x] 2.4 扩展 `validator.py` 和 `evaluator.py`，使 radar-only 评估只准备雷达输入并保存现有 Top-K、DBA、loss 输出。

## 3. 配置与文档

- [x] 3.1 新增 `configs/radar/no_kd.yaml`，设置 `experiment.task: radar`，并将训练主模型配置为 `radar_teacher`。
- [x] 3.2 如需要，新增 `configs/radar/logits_kd.yaml` 或保留后续扩展说明，确保 no-KD 基线不依赖未提供的 RadarTeacher 权重。
- [x] 3.3 更新 README 或扩展指南，说明 radar-only 训练/评估命令，以及 `model.student.type: radar_teacher` 在 no-KD 基线中的含义。

## 4. 测试与验证

- [x] 4.1 新增单元测试，验证 `MODELS.build()` 可构建 `radar_teacher`，并用随机雷达张量检查输出形状。
- [x] 4.2 新增配置测试，验证 radar-only no-KD 配置可加载、任务名为 `radar`、主模型类型为 `radar_teacher`。
- [x] 4.3 使用 `conda run -n kd_mm_beam python -m pytest` 运行相关测试。
- [x] 4.4 使用 `conda run -n kd_mm_beam python scripts/evaluate.py --config configs/radar/no_kd.yaml --weights <radar-teacher-weight>` 做评估烟测；如本地缺少权重或数据集，记录阻塞原因。
- [x] 4.5 运行 `openspec validate add-radar-teacher-model --strict`，确认 change artifacts 合法。

  注：本地缺少 RadarTeacher 权重，`outputs/radar_no_kd/checkpoints/best.pth` 不存在；已额外用随机初始化模型和 `data.dataset.portion=0.001` 验证 radar-only 评估路径可完成。
