## 1. Layout 基础设施

- [x] 1.1 新增 `src/kd_sensing/data/layouts.py` 或等价模块，定义 DeepSense6G 和 MMW 的 dataset family 目录 helper。
- [x] 1.2 为 DeepSense6G layout 提供 `dataset/DeepSense6G/scenario9|scenario31|scenario32` 规范根目录和 `dataset/scenario9|scenario31|scenario32` legacy 根目录。
- [x] 1.3 为 MMW layout 提供 `dataset/MMW/sunny|rainy|foggy/Sensor_Data` 与 `Channel_Data` 目录描述，不实现 MMW loader。

## 2. DeepSense6G 场景路径

- [x] 2.1 更新 `src/kd_sensing/data/scenes.py`，使 DeepSense6G 场景默认 `data_root` 来自 layout helper 并指向 `dataset/DeepSense6G/scenario*`。
- [x] 2.2 保持 `data.dataset.scene` 整数和字符串别名不变，保持旧 `scenario9|scenario31|scenario32` dataset type 拒绝逻辑不变。
- [x] 2.3 增加显式旧 `data_root: dataset/scenario*` 覆盖测试，确认 normalize/retarget 不会把显式旧路径改写为新路径。

## 3. 预处理与配置

- [x] 3.1 更新 `configs/preprocess/*.yaml` 中 DeepSense6G 默认 `data_root` 和 `csv_path` 到 `dataset/DeepSense6G/scenario31`。
- [x] 3.2 更新 `src/kd_sensing/cli/preprocess.py` 的 scene override 逻辑，使覆盖到 Scenario 9/31/32 时使用 layout helper 重建规范 scene root 和 `scenarioXX_RA.csv`。
- [x] 3.3 确认序列 CSV 内模态文件路径仍相对 scene root 解析，不增加 `DeepSense6G` 前缀到 CSV 列值。

## 4. 文档与迁移说明

- [x] 4.1 更新 README 的 DeepSense6G 场景章节，说明推荐目录为 `dataset/DeepSense6G/scenario*`。
- [x] 4.2 增加旧目录迁移说明：移动目录、创建软链接或显式设置 `data.dataset.data_root: dataset/scenario*`。
- [x] 4.3 在文档中记录未来 MMW 目录约定：`dataset/MMW/<sunny|rainy|foggy>/Sensor_Data` 与 `Channel_Data`。

## 5. 测试与验证

- [x] 5.1 更新 `tests/test_training_io_workflow.py` 中默认场景路径、scene override 和显式旧路径兼容断言。
- [x] 5.2 更新 `tests/test_modality_visual_diagnostics.py` 中与 DeepSense6G 默认数据根目录相关的断言和测试 fixture。
- [x] 5.3 添加或更新 layout helper 单元测试，覆盖 DeepSense6G 规范根目录、legacy 根目录和 MMW 条件子目录。
- [x] 5.4 运行 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py tests/test_modality_visual_diagnostics.py -q`。
- [x] 5.5 如相关测试通过，再运行 `conda run -n kd_mm_beam pytest -q` 或记录无法全量运行的原因。
