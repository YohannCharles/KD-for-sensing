## 1. 场景默认值

- [x] 1.1 将 `src/kd_sensing/data/scenes.py` 中默认 DeepSense6G scene 常量改为 31，并确认 Scenario 31 描述符、别名和默认 `dataset/scenario31` 路径可被解析。
- [x] 1.2 将 `src/kd_sensing/config/defaults.py` 的默认 `paths.data_root`、`data.dataset.scene`、`scene_id` 和 `scene_slug` 更新为 Scenario 31。
- [x] 1.3 确认旧 `data.dataset.type: scenario31` 仍作为 removed dataset type 被拒绝，但 `data.dataset.scene: scenario31` 作为场景别名可用。

## 2. 配置与默认路径

- [x] 2.1 更新训练相关 `configs/**/*.yaml` 中默认 `data.dataset.scene: 32` 为 `31`，并将默认 `dataset/scenario32` 预处理路径切到 `dataset/scenario31`。
- [x] 2.2 更新 KD、fusion、canonical 配置和脚本默认的 `outputs/scene32/...` teacher checkpoint、teacher registry、Stage 2/3 checkpoint 路径为 `outputs/scene31/...`。
- [x] 2.3 审查 `src/kd_sensing/config/canonical.py`、`scripts/build_teacher_registry.py` 和相关 registry 工具中的默认路径，确保默认训练依赖指向 `scene31`，显式 Scene 32 仍可通过配置保留。
- [x] 2.4 使用 `rg "scene32|scenario32|scene: 32" configs src scripts README.md tests openspec/specs` 复查剩余引用，并只保留显式 Scene 32、历史产物、跨场景诊断或测试夹具需要的引用。

## 3. 文档与测试

- [x] 3.1 更新 README 中默认 DeepSense6G 场景、默认输出目录、checkpoint/registry 示例和场景覆盖说明。
- [x] 3.2 更新配置解析、训练输出分组、checkpoint registry、canonical 配置和学生/KD 配置测试中的默认断言为 Scenario 31，并保留显式 Scenario 32 覆盖测试。
- [x] 3.3 如缺少覆盖，新增或扩展测试验证未显式设置 `data.dataset.scene` 时解析为 `scene31`，且 `data.dataset.scene=32` 仍解析为 `scene32`。
- [x] 3.4 使用 `conda run -n kd_mm_beam pytest` 运行相关测试；至少覆盖 `tests/test_training_io_workflow.py`、`tests/test_student_configs.py` 以及涉及场景默认值的测试文件。

## 4. OpenSpec 验证

- [x] 4.1 使用 `openspec status --change set-default-training-scene-31` 确认变更 artifact 状态完整。
- [x] 4.2 使用 `openspec validate set-default-training-scene-31 --strict` 验证 delta specs；若 CLI 需要不同参数格式，使用等价的 strict validation 命令。
