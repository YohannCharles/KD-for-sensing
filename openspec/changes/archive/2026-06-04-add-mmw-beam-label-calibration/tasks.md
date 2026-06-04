## 1. Label Mapping Core

- [x] 1.1 新增 MMW beam label calibration 配置解析结构，支持 `enabled`、`label_space`、`num_classes`、`direction`、`offset`、`scene_overrides` 和显式 permutation。
- [x] 1.2 实现中心化 raw↔calibrated mapping helper，覆盖 affine、inverse、distribution reorder、mapping fingerprint 和非法配置错误。
- [x] 1.3 为 mapping helper 添加单元测试，验证 0/63 边界、`direction=-1`、scene override、distribution 重排和 inverse round-trip。

## 2. MMW Dataset Integration

- [x] 2.1 在 MMW/DeepSense6G beam label 读取路径接入 calibration，使 `input_beam` 和 `target_beam` 在启用时返回 calibrated label。
- [x] 2.2 处理显式 `future_beam_label*`、`beam_label` 和 beam power path `argmax` 三类来源，确保 raw label provenance 可写入 metadata。
- [x] 2.3 更新 beam label cache，使缓存值或 cache metadata 区分 mapping fingerprint，避免不同 mapping 复用错误 label。
- [x] 2.4 确保启用 calibration 不改变 `mmwave`、GPS、image、LiDAR、radar、CSI 的按需读取和张量 shape。

## 3. Class Distribution Targets

- [x] 3.1 更新 soft beam label 生成，使 target-domain Gaussian soft label 使用 calibrated hard label 和 calibrated circular topology。
- [x] 3.2 更新 source-domain power/RSS soft label，使 raw power distribution 重排到 calibrated class order。
- [x] 3.3 更新 beamspace physical label 构造和 cache metadata，使 `beamspace_power_label` 按 mapping 重排，并在 mapping mismatch 时拒绝复用或重建缓存。
- [x] 3.4 添加测试覆盖 hard label、soft target、beamspace physical label argmax/class order 一致性。

## 4. Metadata and Diagnostics

- [x] 4.1 在 run metadata、sample metadata、prediction export 和 viewer manifest 中记录 `beam_label_space`、mapping 参数和 mapping fingerprint。
- [x] 4.2 更新 MMW preparation/split metadata，保留 raw beam label provenance，并在启用 calibration 时可输出 calibrated histogram。
- [x] 4.3 更新 beam distribution shift diagnostics，支持 raw/calibrated label space 声明，避免跨 label space 混算距离。
- [x] 4.4 更新 GPS-angle correspondence 和 prediction error label distribution 输出，记录 `beam_index_mode`、mapping 参数和一致的邻近正确率拓扑。

## 5. Configuration and Workflow

- [x] 5.1 为 MMW calibrated label 实验添加推荐配置或 CLI override 示例，默认保持 raw label space。
- [x] 5.2 确保旧 raw-label 配置、旧 checkpoint 评估和非 MMW dataset 不需要 calibration 字段即可运行。
- [x] 5.3 在相关 README 或实验说明中记录 raw/calibrated label space 的比较边界和旧结果不可直接混比的注意事项。

## 6. Validation

- [x] 6.1 运行 `openspec validate add-mmw-beam-label-calibration --strict` 并修复所有 spec 问题。
- [x] 6.2 运行 `openspec status --change add-mmw-beam-label-calibration` 确认 proposal、design、specs 和 tasks 均完成。
- [x] 6.3 运行 `conda run -n kd_mm_beam pytest tests/test_mmwave_modality.py tests/test_modality_visual_diagnostics.py -q` 验证 MMW dataset、mmWave 输入和诊断兼容。
- [x] 6.4 运行 `conda run -n kd_mm_beam pytest tests/test_hist_beam_v8_target_prior.py tests/test_v7_shared_physical_private_residual.py tests/test_history_anchored_residual_beam.py -q` 验证 soft label、physical label 和 history-anchor 路径。
- [x] 6.5 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py tests/test_student_configs.py -q` 验证配置和架构边界未回退。
