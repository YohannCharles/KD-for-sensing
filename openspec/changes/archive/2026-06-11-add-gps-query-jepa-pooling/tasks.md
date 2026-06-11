## 1. 模型与 encoder 支撑

- [x] 1.1 在 `src/kd_sensing/models/jepa.py` 新增 `GPSQueryPool`，支持 `[B,T,N,D]` patch tokens、`[B,T,C]` 条件特征、`k_queries`、`num_heads`、dropout、输出 `[B,T,D]` 和可选 attention map。
- [x] 1.2 扩展 `JepaContextImageEncoder` 配置解析，保留默认 `pooling: mean`，新增显式 `pooling: gps_query_attention`、`gps_query_pool` 参数、训练策略 metadata 和缺失条件特征报错。
- [x] 1.3 为 `JepaContextImageEncoder.forward()` 增加 GPS condition feature 参数，在 GPS-query pooling 下调用 `GPSQueryPool`，在 mean pooling 下保持现有单 image input 行为。
- [x] 1.4 更新 `__all__`、注册表导入或相关类型标注，确保 `GPSQueryPool` 和扩展后的 `jepa_context_image` 可被 focused tests 构建。

## 2. 模块化序列模型条件调用

- [x] 2.1 在 `ModularSequenceModel.forward()` 中实现 dependency-aware encoder 调用：普通 encoder 单输入调用，声明条件依赖的 encoder 接收已满足的条件 feature。
- [x] 2.2 为 GPS-query image encoder 支持 projected GPS condition source，确保 GPS encoder 和 projector 输出 `[B,T,d_model]` 后传给 image encoder。
- [x] 2.3 增加缺失条件模态、循环或无法满足依赖、batch/time 不一致时的清晰错误信息。
- [x] 2.4 验证现有 image、GPS、LiDAR、mmWave、CSI、coord、ray 等未声明条件依赖的配置无需新增字段即可保持 forward 兼容。

## 3. 配置、metadata 与文档

- [x] 3.1 基于 `configs/fusion/experiments/jepa_image_gps/image_gps_jepa_gps_biased_best_beambench_fair_lowmem.yaml` 新增 GPS-query pooling 派生配置，复用 GPS-biased 多场景 JEPA checkpoint 并更新 run name。
- [x] 3.2 如需要同步 2604-style S32/S33/S34 macro 口径，基于对应 `fair_gps_biased` 配置新增同参数派生配置，并避免引用 scene31-only checkpoint。
- [x] 3.3 更新 `src/kd_sensing/engine/run_metadata.py`，在 JEPA downstream metadata 中记录 pooling、GPS-query 是否启用、`k_queries`、`num_heads`、condition source、checkpoint path 和 freeze 状态。
- [x] 3.4 更新 `configs/fusion/experiments/jepa_image_gps/README.md`，说明 mean-pooling `fair_gps_biased` baseline 与 GPS-query pooling 派生配置的关系和推荐比较口径。

## 4. 测试

- [x] 4.1 在 JEPA focused tests 中新增 `GPSQueryPool` shape、K-query 平均、attention map 诊断和条件维度校验用例。
- [x] 4.2 新增 `JepaContextImageEncoder` mean 默认兼容、GPS-query pooling forward、缺失 GPS condition feature 报错和 checkpoint 加载兼容用例。
- [x] 4.3 新增 `ModularSequenceModel` image+GPS GPS-query pooling synthetic forward smoke，验证 projected GPS condition source、logits shape 和 `encoder_features`/`modality_features` 输出。
- [x] 4.4 扩展 JEPA downstream 配置加载与 runtime metadata 测试，覆盖新 GPS-query 配置、checkpoint 路径、pooling metadata 和不引用 retired 路线。

## 5. 验证

- [x] 5.1 运行 `openspec validate add-gps-query-jepa-pooling --strict` 并修复所有 OpenSpec 问题。
- [x] 5.2 运行 `conda run -n kd_mm_beam pytest tests/test_gps_conditioned_jepa.py -q`。
- [x] 5.3 若修改影响模块化 core 或 shared forward，运行 `conda run -n kd_mm_beam pytest tests/test_modular_sequence_next_query_transformer.py tests/test_architecture_boundaries.py -q`。
- [x] 5.4 视改动范围运行 `conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_config_load_characterization.py -q`，确认配置与入口没有回归。
