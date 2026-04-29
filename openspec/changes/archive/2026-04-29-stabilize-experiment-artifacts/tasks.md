## 1. Registry 与配置基础

- [x] 1.1 在默认配置中新增最佳 checkpoint registry 配置项，覆盖 registry 目录、是否启用、是否优先加载、归档 metric 和归档文件名格式。
- [x] 1.2 新增 `kd_sensing.utils.artifact_registry` 或等价模块，实现 slug 推导、精度格式化、sidecar metadata 读写、同 slug 候选筛选和最高验证 Top-1 checkpoint 解析。
- [x] 1.3 为 registry 工具补充单元测试，使用 `conda run -n kd_mm_beam pytest ...` 验证命名、旧候选清理、metadata 读写和解析优先级。

## 2. 训练流程集成

- [x] 2.1 修改训练循环，独立跟踪最高验证 Top-1 accuracy，并在刷新或训练结束时保存/复制对应 checkpoint 到 registry。
- [x] 2.2 将 registry 目录、归档 checkpoint、源 checkpoint、验证 Top-1、epoch、split metadata 和归一化工件路径写入 `train_log.json` 与最终运行配置。
- [x] 2.3 为 `GPSStandardScaler` 增加 save/load，并在启用 GPS 归一化训练时保存 `artifacts/gps_scaler.npz`。
- [x] 2.4 在启用 LiDAR streaming stats 归一化训练时保存 `artifacts/lidar_normalizer.npz`，并继续支持用户显式提供的 stats 文件。

## 3. KD 与评估加载

- [x] 3.1 修改 KD teacher checkpoint 解析逻辑，按显式路径、registry、旧 `paths.weights_dir / teacher_model_name` 的顺序加载，并记录来源。
- [x] 3.2 修改评估入口，使 `--weights` 或绝对路径保持最高优先级；默认权重解析可复用 registry，并将来源写入 `test_report.json`。
- [x] 3.3 修改 GPS/LiDAR 评估数据集构建逻辑，优先从 checkpoint metadata 或 registry sidecar 加载训练归一化工件，只有缺失 metadata 时才使用当前兼容回退。
- [x] 3.4 为 KD teacher 加载、评估加载和归一化工件复用补充测试，使用 `conda run -n kd_mm_beam pytest ...` 运行目标测试。

## 4. 数据与预处理语义

- [x] 4.1 修正 Scenario 9 序列生成窗口条件，确保每个 `seq_index` 的最后一个合法窗口被生成，并补充窗口数量测试。
- [x] 4.2 修改 `portion < 1.0` 采样策略，默认使用确定性且覆盖全局分布的采样方式；运行 metadata 记录采样策略、seed、样本数和 `seq_index` 范围。
- [x] 4.3 删除未使用的 `gps_smooth_window` 代码和配置入口，包括默认/示例配置字段、`Scenario9Dataset` 显式参数、`load_gps_feature_sequence`/`build_gps_features` 参数传递、README 文档引用，并补充测试确认 GPS `relative_polar` 输出不受历史字段影响。
- [x] 4.4 更新相关数据加载测试，使用 `conda run -n kd_mm_beam pytest ...` 验证模态选择、portion、GPS 和 LiDAR 路径兼容。

## 5. 文档与规格一致性

- [x] 5.1 更新 README 或实验说明，说明 registry 默认目录、文件命名、加载优先级、归一化工件和旧路径回退行为。
- [x] 5.2 检查并更新默认配置注释或示例，确保 GPS/Radar/LiDAR 单模态 teacher/student/KD `gru_params` 与 `[64, 64, 1]` 规格一致。
- [x] 5.3 运行 `openspec validate --all`，确认 change specs 和现有 specs 均通过校验。

## 6. 端到端验证

- [x] 6.1 运行 `conda run -n kd_mm_beam pytest -q -p no:cacheprovider`，确认完整测试通过。
- [x] 6.2 使用 `conda run -n kd_mm_beam python scripts/train.py --config <small-or-synthetic-config> --override training.epochs=1 ...` 做短训练 smoke test，确认 registry checkpoint、sidecar、训练日志和归一化工件被写出。
- [x] 6.3 使用 `conda run -n kd_mm_beam python scripts/evaluate.py --config <matching-config>` 做评估 smoke test，确认默认可从 registry 加载 checkpoint 并复用训练归一化工件。
