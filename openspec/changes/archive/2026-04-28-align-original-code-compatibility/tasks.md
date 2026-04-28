## 1. 参数映射与配置对齐

- [x] 1.1 整理 `All_models/params_Image*.txt`、`All_models/params_Both*.txt`、`/tmp/KD-for-sensing-original/train_image.py` 和 `/tmp/KD-for-sensing-original/train_both.py` 的参数映射表，记录每个 image 单模态、其它单模态继承项与 image+radar 配置应使用的 GRU 层数、batch size、seed、lr、weight_decay、KD 参数、scheduler 和 early stopping 参数。
- [x] 1.2 更新 `src/kd_sensing/config/defaults.py`，取消所有 teacher/student 默认统一二层 GRU 的假设，并让共享默认值不覆盖显式原代码兼容配置。
- [x] 1.3 更新 `configs/image/teacher_no_kd.yaml`、`configs/image/student_no_kd.yaml`、`configs/image/logits_kd.yaml` 和 `configs/image/rkd.yaml`，使 image teacher/student 使用 `[64, 64, 1]` 并对齐对应复现实验超参数。
- [x] 1.4 更新 `configs/radar/*.yaml`、`configs/gps/*.yaml` 和 `configs/lidar/*.yaml` 中的单模态配置，使 teacher/student 使用 `[64, 64, 1]`，并让共享训练字段、scheduler 字段和 KD 字段与 `configs/image/` 下同角色配置一致。
- [x] 1.5 更新 `configs/fusion/no_kd.yaml`、`configs/fusion/logits_kd.yaml`、`configs/fusion/rkd.yaml` 和 `configs/fusion/image_radar_*.yaml`，使 image+radar teacher 使用 `[64, 64, 2]`、student 使用 `[64, 64, 1]` 并对齐对应复现实验超参数。
- [x] 1.6 检查非 image+radar 的 fusion 扩展配置，确保文档和测试不把这些扩展配置误标为原代码/原论文参数。

## 2. Checkpoint 加载与恢复训练

- [x] 2.1 在 checkpoint 工具中实现统一的可诊断加载函数，默认 `strict=True`，错误信息包含 checkpoint 路径、模型角色、missing keys 和 unexpected keys。
- [x] 2.2 将 KD teacher 加载和评估入口改为使用统一加载函数，移除静默 `strict=False`，并支持显式非严格加载时记录 mismatch keys。
- [x] 2.3 实现 `training.resume`：`true` 从 `output.run_name/checkpoints/last.pth` 恢复，路径字符串从显式 checkpoint 恢复，并在路径不存在时提前报错。
- [x] 2.4 恢复 checkpoint 时加载 student 模型、optimizer、scheduler、epoch 和 best validation loss；保存 `last.pth` 时补齐后续恢复所需字段。
- [x] 2.5 确认 `training.start_epoch` 仅在 checkpoint 缺少 epoch 字段时作为兜底，不再替代真实 resume 逻辑。

## 3. 固定输入尺寸校验

- [x] 3.1 为目标兼容 image-only 与包含 image 的 fusion 配置增加 image size 校验，拒绝非 `[224, 224]` 的 `data.dataset.image_size`。
- [x] 3.2 为目标兼容 radar-only 与包含 radar 的 fusion 配置增加 radar RA/DA size 校验，拒绝非 `128x64` 的 radar 输入尺寸配置或 synthetic 测试尺寸。
- [x] 3.3 修正 image motion mask 路径的尺寸来源或校验位置，避免配置写了其它 image size 但运行时仍硬编码 `224x224`。
- [x] 3.4 更新错误信息，使其明确指出当前限制来自 image/fusion teacher FC 输入、motion mask 或 radar branch 结构。

## 4. 测试覆盖

- [x] 4.1 更新配置测试，断言 image/radar/GPS/LiDAR 单模态均为一层 GRU、其它单模态共享字段与 image 同角色配置一致、image+radar teacher 为二层 GRU、image+radar student 为一层 GRU，并验证关键复现实验超参数。
- [x] 4.2 添加 checkpoint mismatch 测试，用一层 GRU 权重加载二层 GRU 配置时必须报出 missing GRU layer-1 keys。
- [x] 4.3 添加评估入口权重加载测试，验证结构不匹配时不再静默通过，显式非严格加载时会记录 missing/unexpected keys。
- [x] 4.4 添加 resume 训练测试，使用小型 synthetic 或 fixture 配置验证 `conda run -n kd_mm_beam pytest <test-file>` 下能恢复模型、optimizer、scheduler 和 epoch。
- [x] 4.5 添加 image/radar 固定尺寸校验测试，覆盖非法 image size 和非法 radar size 的错误信息。

## 5. 文档与验证

- [x] 5.1 更新 `README.md` 和 `docs/extension_guide.md`，说明原代码兼容与单模态一致性配置矩阵、GRU 层数差异、随附权重兼容策略和固定输入尺寸限制。
- [x] 5.2 更新或新增与复现相关的说明，明确 radar-only、GPS-only 和 LiDAR-only 单模态参数继承 image 单模态，非 image+radar fusion 仍是项目扩展配置，不作为上游原代码参数声明。
- [x] 5.3 使用 `conda run -n kd_mm_beam pytest` 运行全量测试，并记录结果。
- [x] 5.4 使用 `conda run -n kd_mm_beam python scripts/train.py --config <small-original-compatible-config> ...` 运行短训练 smoke test，验证 forward、loss、backward、validation 和 checkpoint 保存路径。
- [x] 5.5 使用 `conda run -n kd_mm_beam python scripts/evaluate.py --config <original-compatible-config> --weights <matching-weight>` 验证匹配权重可严格加载并完成评估。
