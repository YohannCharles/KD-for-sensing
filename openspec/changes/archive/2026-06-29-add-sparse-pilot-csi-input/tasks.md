## 1. OpenSpec 与配置

- [x] 1.1 新增 sparse pilot CSI 输入模式的 OpenSpec delta，并校验 change。
- [x] 1.2 新增 `physics_informed_mmw_sparse_pilot_multimodal.yaml` 配置，保留现有入口。

## 2. Adapter 实现

- [x] 2.1 扩展 `PhysicsAdapterConfig`，支持 pilot stride、pattern 和 seed。
- [x] 2.2 实现 `csi_input_mode=sparse_pilot`，输出 `csi_input`、`csi_observation_mask` 和 metadata。
- [x] 2.3 保持 `csi_target` 不进入 model forward，`oracle_full` guard 不变。

## 3. 验证

- [x] 3.1 补充 focused tests，覆盖 sparse pilot mask、未观测位置置零和配置加载。
- [x] 3.2 运行 `conda run -n kd_mm_beam pytest tests/test_physics_informed_mmw.py -q`。
- [x] 3.3 运行 `openspec validate add-sparse-pilot-csi-input --strict`。
