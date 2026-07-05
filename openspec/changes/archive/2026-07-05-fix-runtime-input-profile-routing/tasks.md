## 1. Runtime 修复

- [x] 1.1 检查 `src/kd_sensing/engine/runtime.py` 中所有单模态 `prepare_task_inputs` 分支的 profile key。
- [x] 1.2 将 radar/gps/lidar/mmwave/csi 分支统一改为读取同名 `model_cfg.input_profiles.<modality>`。
- [x] 1.3 确认缺省 profile 行为仍由对应 helper 或 modality contract 处理。

## 2. Focused tests

- [x] 2.1 新增或扩展 runtime focused test，覆盖 radar/gps/lidar 的 profile 透传。
- [x] 2.2 覆盖 `input_profiles` 缺省时不读取其它 modality profile 的场景。
- [x] 2.3 如测试触碰 mmwave/csi helper，使用 synthetic tensor，不读取真实 `dataset/`。

## 3. 验证

- [x] 3.1 运行 `conda run -n kd_mm_beam pytest <runtime-profile-focused-test> -q`。
- [x] 3.2 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [x] 3.3 如修改配置解析或 modality contract，追加 `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py -q`。（本次未修改配置解析或 modality contract，未触发追加验证。）
