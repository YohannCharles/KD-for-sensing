## 1. 契约和测试基线

- [x] 1.1 运行 `openspec validate modularize-jepa-downstream-extensibility --strict`，修复 proposal/design/spec delta 的格式和 requirement 问题。
- [x] 1.2 新增或扩展 JEPA downstream focused tests，先覆盖 mean pooler 兼容、GPS-query pooler 构建、identity adapter、未知 pooler/adapter 错误和 `jepa_context_image` synthetic forward。
- [x] 1.3 新增或扩展 modular sequence tests，覆盖 projected、encoded、raw 条件特征来源、缺失依赖、batch/time mismatch、循环依赖和普通 encoder 兼容。
- [x] 1.4 新增 optimizer 参数组 tests，覆盖命名参数组、未匹配 pattern、重复匹配、默认单 `main` 组兼容和 optimizer summary。
- [x] 1.5 新增 runtime metadata tests，覆盖模型声明 metadata、config fallback、pooler/adapter 字段、checkpoint 字段和参数组摘要。

## 2. JEPA downstream pooler/adapter 边界

- [x] 2.1 新建或整理 JEPA downstream 窄模块，承载 mean pooler、GPS-query attention pooler、identity adapter 和相关 shape/metadata helper。
- [x] 2.2 增加 JEPA downstream pooler/adapter 构建入口，支持 `pooler: {type: ...}` 和 `adapter: {type: ...}` 配置。
- [x] 2.3 保留 `pooling: mean` 和 `pooling: gps_query_attention` 作为兼容 alias，并将其规范化为等价 pooler 配置。
- [x] 2.4 重构 `JepaContextImageEncoder`，使其只负责 checkpoint context encoder 加载、patch token 生成、pooler/adapter 调用、freeze 和 metadata 暴露。
- [x] 2.5 确保 `JepaContextImageEncoder` 默认输出仍为 `[B,T,D]`，且现有 `fair_gps_biased` 和 GPS-query 配置无需迁移即可 forward。

## 3. conditioned encoder 契约

- [x] 3.1 整理 `ModularSequenceModel` 的 encoder 条件依赖 helper，正式支持 `required_context_modalities`、`context_feature_source` 和 `context_feature_kwargs`。
- [x] 3.2 扩展条件来源处理，支持 projected、encoded 和显式声明的 raw condition feature。
- [x] 3.3 改进依赖排序和错误信息，确保缺失模态、自依赖、循环依赖和无法满足依赖时报告 pending modalities 与 unmet dependencies。
- [x] 3.4 补充 batch/time shape 校验，确保条件特征与目标 encoder 原始输入不一致时不静默广播或截断。
- [x] 3.5 确认未声明依赖的 image、radar、GPS、LiDAR、mmWave、CSI、coord 和 ray encoder 保持单输入调用兼容。

## 4. optimizer 参数组

- [x] 4.1 在 `kd_sensing.engine.optim` 中实现 `training.optimizer.parameter_groups` 解析，支持命名组、module pattern、lr、weight_decay 和可选 strict/require_all_matched 行为。
- [x] 4.2 实现 trainable parameter 匹配和去重逻辑，拒绝或诊断重复匹配、未匹配 pattern 和无 trainable 参数组。
- [x] 4.3 保持未声明 parameter groups 时的现有 Adam 单 `main` 组行为。
- [x] 4.4 扩展 `optimizer_param_group_summary()` 或等价 helper，记录每组 `name`、`lr`、`weight_decay` 和 `param_count`。
- [x] 4.5 确认 `engine.trainer` 只消费 optimizer 和 summary，不新增 JEPA 专属参数组分支。

## 5. metadata 和配置

- [x] 5.1 为 `ModularSequenceModel`、`JepaContextImageEncoder`、pooler/adapter 或相关子模块提供只读 `training_strategy_metadata()` 或等价 metadata 方法。
- [x] 5.2 扩展 `engine.run_metadata` 或 artifact writer，优先聚合模型声明 metadata，并保留 config fallback 兼容历史配置。
- [x] 5.3 确保 metadata 记录 JEPA checkpoint、state dict prefix、pooler type、adapter type、condition source、freeze 状态、attention diagnostics 开关、ablation 和 optimizer 参数组摘要。
- [x] 5.4 新增 `fair_gps_biased` 派生配置示例，用于比较 pooler/adapter 或参数组策略，并确保只覆盖实验变量。
- [x] 5.5 更新 `configs/fusion/experiments/jepa_image_gps/README.md`，说明 baseline、pooler/adapter 派生实验、参数组调参和匹配比较口径。

## 6. 验证

- [x] 6.1 运行 `openspec validate modularize-jepa-downstream-extensibility --strict`。
- [x] 6.2 运行 `conda run -n kd_mm_beam pytest tests/test_gps_conditioned_jepa.py -q`，验证 JEPA context reuse、pooling 和下游配置。
- [x] 6.3 运行 `conda run -n kd_mm_beam pytest tests/test_modular_sequence_next_query_transformer.py -q`，验证 modular sequence core 与 conditioned encoder 兼容。
- [x] 6.4 运行新增或扩展的 optimizer/metadata focused tests，例如 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py -q` 中相关用例或新建 focused test 文件。
- [x] 6.5 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py tests/test_config_load_characterization.py -q`，确认导入边界、配置加载和退役入口 guardrail 无回归。
- [x] 6.6 如实现触碰 CLI、README 推荐命令或实验配置路径，追加运行 `conda run -n kd_mm_beam pytest tests/test_cli_help.py -q`。
