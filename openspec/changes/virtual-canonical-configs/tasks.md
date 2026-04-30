## 1. Canonical Resolver

- [x] 1.1 新增或扩展配置解析模块，定义 canonical 模态顺序 `image, radar, gps, lidar, mmwave`、支持的 fusion mode 后缀和路径匹配规则。
- [x] 1.2 实现 `configs/fusion/<slug>_<mode>.yaml` 的虚拟配置生成逻辑，输出相对 `DEFAULT_CONFIG` 的 override dict。
- [x] 1.3 为 slug 校验实现清晰错误：乱序、重复模态、未知模态、单模态 fusion slug、未知 mode 和非 fusion 缺失路径。
- [x] 1.4 将 resolver 集成到 `load_config()`，确保实体 YAML 优先、虚拟配置次之、CLI override 最后应用。
- [x] 1.5 保持 image+radar special case 的上游兼容参数，包括 GRU 层数、训练/KD 参数和 `All_models/BothTeacher_best.pth` 默认 teacher 来源。
- [x] 1.6 为包含 GPS、LiDAR、mmWave 的虚拟配置补齐 dataset 和模型输入字段，确保 teacher/student `modalities` 顺序一致。

## 2. Tests

- [x] 2.1 新增 resolver 单元测试，覆盖合法 canonical 路径、缺失非 canonical 文件、实体 YAML 优先和 CLI override 优先级。
- [x] 2.2 新增命名校验测试，覆盖乱序 slug、重复模态、未知模态、单模态 fusion slug 和错误提示中的 canonical slug。
- [x] 2.3 更新 `tests/test_student_configs.py`，把 canonical fusion 覆盖从 `Path.exists()` 改为 `load_config()` 语义验证。
- [x] 2.4 更新 `tests/test_gps_modality.py`、`tests/test_lidar_modality.py` 和 `tests/test_mmwave_modality.py`，确保它们枚举的 canonical fusion path 可虚拟加载。
- [x] 2.5 在删除实体 canonical YAML 前，用 `conda run -n kd_mm_beam pytest tests/test_student_configs.py tests/test_gps_modality.py tests/test_lidar_modality.py tests/test_mmwave_modality.py` 验证生成配置与现有测试契约一致。

## 3. Config Cleanup and Docs

- [x] 3.1 删除 `configs/fusion/` 下可由 resolver 生成的 26 × 4 个 canonical YAML 文件。
- [x] 3.2 保留 legacy alias YAML：`no_kd.yaml`、`logits_kd.yaml`、`rkd.yaml`、`image_gps_no_kd.yaml`、`radar_gps_no_kd.yaml`、`radar_lidar_no_kd.yaml`、`all_modalities_no_kd.yaml` 和 `all_modalities_lidar_no_kd.yaml`。
- [x] 3.3 更新 README，说明 canonical fusion 配置路径可以是虚拟配置，并明确命名顺序为 `image > radar > gps > lidar > mmwave`。
- [x] 3.4 更新 `docs/extension_guide.md`，说明实体 YAML 优先、缺失 canonical fusion path 由 loader 生成，以及新实验应使用 canonical path。

## 4. Verification

- [x] 4.1 运行 `conda run -n kd_mm_beam pytest tests/test_student_configs.py tests/test_gps_modality.py tests/test_lidar_modality.py tests/test_mmwave_modality.py tests/test_training_io_workflow.py`。
- [x] 4.2 运行 `conda run -n kd_mm_beam python scripts/train.py --config configs/fusion/gps_mmwave_logits_kd.yaml --dry-run`，确认删除实体 YAML 后虚拟配置仍能启动训练入口。
- [x] 4.3 运行 `openspec status --change virtual-canonical-configs`，确认变更 artifacts 和任务状态可进入实施阶段。
