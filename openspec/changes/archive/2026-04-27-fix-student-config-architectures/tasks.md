## 1. 配置修正

- [x] 1.1 将 `configs/image/no_kd.yaml`、`configs/image/logits_kd.yaml`、`configs/image/rkd.yaml` 的 `model.student.type` 从 `image_teacher` 改为 `image_student`。
- [x] 1.2 将 `configs/fusion/no_kd.yaml`、`configs/fusion/logits_kd.yaml`、`configs/fusion/rkd.yaml` 的 `model.student.type` 从 `fusion_teacher` 改为 `fusion_student`。
- [x] 1.3 将所有 fusion student 配置的 `gru_params` 修正为 `[64, 64, 1]`，保持 fusion teacher 配置为 `[64, 64, 2]`。
- [x] 1.4 将 `src/kd_sensing/config/defaults.py` 中默认 `model.student.type` 修正为 `image_student`。

## 2. 回归检查

- [x] 2.1 新增或更新配置构建测试，验证 image 默认配置构建 `ImageStudentModalityNet`，fusion 默认配置构建 `StudentModalityNet`。
- [x] 2.2 新增或更新权重兼容性检查，使用 `conda run -n kd_mm_beam` 验证默认 student 配置与对应 `All_models/*Std*.pth` 权重没有 missing key 或 shape mismatch。
- [x] 2.3 确认 `total_ops`、`total_params` 等非模型统计项不会被误判为权重结构不兼容。

## 3. 验证

- [x] 3.1 使用 `conda run -n kd_mm_beam python -m pytest` 运行相关测试。
- [x] 3.2 使用 `conda run -n kd_mm_beam python scripts/evaluate.py --config configs/image/no_kd.yaml --weights All_models/ImageStd_noKD.pth` 做 image student 权重加载烟测；如缺少数据集则记录阻塞原因。
- [x] 3.3 使用 `conda run -n kd_mm_beam python scripts/evaluate.py --config configs/fusion/no_kd.yaml --weights All_models/BothStd_noKD.pth` 做 fusion student 权重加载烟测；如缺少数据集则记录阻塞原因。
- [x] 3.4 运行 `openspec validate fix-student-config-architectures --strict`，确认 change artifacts 合法。
