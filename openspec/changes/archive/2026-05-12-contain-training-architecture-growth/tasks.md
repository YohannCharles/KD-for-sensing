## 1. 行为锁定与边界基线

- [x] 1.1 增加 trainer characterization 测试，锁定当前 history keys、epoch_log scalar keys、checkpoint payload keys 和 `final_config.yaml` 写出语义。
- [x] 1.2 增加 G2D characterization 测试，锁定 `G2DDistiller.compute()` loss 字段、diagnostics 字段、SMP active modalities 和 epoch diagnostics 文件语义。
- [x] 1.3 增加 CRAF/MARF extra loss characterization 测试，覆盖 beam soft、unimodal aux、counterfactual、reliability KD、MARF subset training 和 scalar diagnostics key。
- [x] 1.4 使用 `conda run -n kd_mm_beam pytest tests/test_g2d_distiller.py tests/test_g2d_loss.py tests/test_g2d_smp.py tests/test_craf_fusion.py tests/test_marf_fusion.py tests/test_marf_training.py -q` 验证迁移前行为基线。

## 2. 共享 forward runtime

- [x] 2.1 提取共享 task forward helper，统一 batch 标准化、labels 准备、模态输入准备、model forward、`ModelOutput` 适配和 future slot 选择。
- [x] 2.2 将 `engine.trainer` 的 `_forward_for_task()` 调用点切换到共享 runtime helper，并保持 AMP、force modality mask、reliability gate 和 gate temperature 行为不变。
- [x] 2.3 将 `engine.validator` 的 task 分支切换到共享 runtime helper，并验证 official metrics 与 modality subset validation 行为不变。
- [x] 2.4 将 viewer prediction 和 G2D teacher runtime 中重复的输入准备/forward 分支切换到共享 runtime helper。
- [x] 2.5 使用 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py tests/test_g2d_distiller.py tests/test_modality_visual_diagnostics.py tests/test_architecture_boundaries.py -q` 验证共享 runtime。

## 3. 训练扩展框架

- [x] 3.1 新增训练扩展接口、extension context、batch state、loss bundle 和 epoch diagnostics 聚合结构，先实现 no-op extension。
- [x] 3.2 调整 `engine.trainer.train()`，让主循环通过扩展点执行 `setup`、`before_epoch`、`after_forward`、`after_backward` 和 `after_epoch` 等生命周期步骤。
- [x] 3.3 保持普通 no-KD、logits KD、RKD 训练路径行为不变，并确认 distiller 参数组、AMP、grad clip、scheduler、early stopping 和 checkpoint 保存仍由主循环统一处理。
- [x] 3.4 使用 `conda run -n kd_mm_beam pytest tests/test_student_configs.py tests/test_training_io_workflow.py tests/test_g2d_loss.py -q` 验证基础训练路径。

## 4. 方法扩展迁移

- [x] 4.1 将 G2D teacher ensemble 构建、checkpoint 解析、teacher forward 和 G2D diagnostics accumulator 接入迁移到 G2D training extension。
- [x] 4.2 调整 `distillation/g2d.py` 和相关导入，使 G2D 算法模块不再依赖 model builder、checkpoint registry、batch preparation 或 teacher runtime。
- [x] 4.3 将 CRAF beam soft、unimodal aux、counterfactual gate、prior regularization、reliability KD 和 CRAF scalar diagnostics 迁移到 CRAF training extension。
- [x] 4.4 将 MARF residual norm、anchor prior regularization、anchor entropy、subset CE/KD 和 MARF scalar diagnostics 迁移到 MARF training extension。
- [x] 4.5 从 `engine.trainer` 删除已迁移的大段方法私有 helper，仅保留扩展选择、调用和通用日志合并逻辑。
- [x] 4.6 使用 `conda run -n kd_mm_beam pytest tests/test_g2d_distiller.py tests/test_g2d_diagnostics.py tests/test_craf_fusion.py tests/test_marf_fusion.py tests/test_marf_training.py tests/test_teacher_prior_craf.py -q` 验证方法扩展迁移。

## 5. 诊断可视化真实拆分

- [x] 5.1 将 `VisualizationConfig`、parse/final config snapshot 和 metadata path 配置逻辑迁移到 `diagnostics.visualization.config`。
- [x] 5.2 将 diagnostic dataset 构建、scene metadata 和 selected CSV frame 逻辑迁移到 `diagnostics.visualization.datasets`。
- [x] 5.3 将 candidate collection、sample selection 和 sampling summary 逻辑迁移到 `diagnostics.visualization.sampling`。
- [x] 5.4 将 tensor/modality/split statistics 逻辑迁移到 `diagnostics.visualization.stats`。
- [x] 5.5 将 sample record、PNG/render 和 processed asset 输出逻辑迁移到 `diagnostics.visualization.render`。
- [x] 5.6 将 JSON、JSONL、CSV 和 summary 写出逻辑迁移到 `diagnostics.visualization.writers`。
- [x] 5.7 保留 `diagnostics.modality_visualization.visualize_modalities` 和 `diagnostics.visualization.core` 的公开兼容入口。
- [x] 5.8 使用 `conda run -n kd_mm_beam pytest tests/test_modality_visual_diagnostics.py tests/test_gradio_complementarity_explorer.py tests/test_architecture_boundaries.py -q` 验证诊断拆分。

## 6. 高级 fusion 配置 overlay

- [x] 6.1 设计并实现高级 fusion 配置 overlay 解析或生成器，支持 base、method、ablation 和 scene override 组合。
- [x] 6.2 为 G2D lite/global/horizon 提供 overlay 入口，并保持现有实体 YAML 可加载。
- [x] 6.3 为 CRAF 和 MARF baseline/ablation 提供 overlay 入口，并确保 ablation overlay 只表达差异字段。
- [x] 6.4 保持实体 YAML 优先级，确保已有 `configs/fusion/*.yaml` 路径不会被虚拟 overlay 覆盖。
- [x] 6.5 扩展配置矩阵测试，覆盖 overlay 入口、实体 YAML 优先级、关键字段一致性和 `final_config.yaml` 完整写出。
- [x] 6.6 使用 `conda run -n kd_mm_beam pytest tests/test_student_configs.py tests/test_g2d_smp.py tests/test_marf_fusion.py tests/test_craf_fusion.py -q` 验证配置 overlay。

## 7. 架构边界与文档

- [x] 7.1 扩展 `tests/test_architecture_boundaries.py`，检查 `engine.trainer` 不再包含 G2D/CRAF/MARF 大段方法私有 helper，新增方法通过扩展模块接入。
- [x] 7.2 扩展架构边界测试，检查 `distillation.g2d` 不导入 model builder、checkpoint registry、batch preparation 或 teacher runtime。
- [x] 7.3 扩展架构边界测试，检查 `diagnostics.visualization.core` 不再作为 config、datasets、sampling、stats、render 和 writers 的主要实现聚合文件。
- [x] 7.4 更新 README 或 `docs/extension_guide.md`，说明训练扩展点、共享 forward runtime、诊断拆分和高级 fusion overlay 的新增开发规则。
- [x] 7.5 使用 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` 验证架构边界。

## 8. 最终验收

- [x] 8.1 使用 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py tests/test_student_configs.py tests/test_g2d_distiller.py tests/test_g2d_loss.py tests/test_g2d_smp.py tests/test_g2d_diagnostics.py tests/test_craf_fusion.py tests/test_marf_fusion.py tests/test_marf_training.py tests/test_modality_visual_diagnostics.py tests/test_architecture_boundaries.py -q` 运行重点回归。
- [x] 8.2 使用 `conda run -n kd_mm_beam pytest -q` 执行全量回归。
- [x] 8.3 使用 `openspec validate --changes contain-training-architecture-growth` 验证 OpenSpec 变更。
- [x] 8.4 使用 `openspec status --change contain-training-architecture-growth` 确认变更 apply-ready。
