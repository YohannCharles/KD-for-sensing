## 1. 基线与 characterization

- [ ] 1.1 确认 `add-raymobtime-s008-selection` 是否需要先归档；若继续保留 active change，记录本变更只处理训练/配置编排边界。
- [ ] 1.2 增加 CLI help focused test，覆盖 `conda run -n kd_mm_beam kd-sensing-train --help`、`kd-sensing-evaluate --help`、`kd-sensing-preprocess --help`、`kd-sensing-export-viewer-manifest --help` 和 `kd-sensing-visualize-modalities --help`。
- [ ] 1.3 增加 config load characterization tests，覆盖实体 YAML、virtual canonical fusion、snapshot virtual 配置、Raymobtime 配置和命令行覆盖顺序。
- [ ] 1.4 增加训练短流程 characterization test，使用 synthetic 或现有 fixture 完成 forward、loss、backward、validation、checkpoint 和 artifact 写出，并校验关键输出字段。
- [ ] 1.5 增加 checkpoint/TensorBoard/history 字段回归断言，锁定 `train_log.json`、`training_outputs.npz`、checkpoint sidecar 和 legacy TensorBoard tag 兼容语义。

## 2. 可视化兼容 CLI 收口

- [ ] 2.1 新增或恢复 `src/kd_sensing/cli/visualize_modalities.py`，实现为委托 `kd_sensing.cli.export_viewer_manifest.main` 的薄 alias。
- [ ] 2.2 更新 `pyproject.toml` 的 `[project.scripts]`，声明 `kd-sensing-visualize-modalities = "kd_sensing.cli.visualize_modalities:main"`。
- [ ] 2.3 更新 README 或 `tools/visualization/README.md`，说明推荐入口是 `kd-sensing-export-viewer-manifest`，兼容入口只导出 manifest。
- [ ] 2.4 扩展架构边界测试，验证 `kd-sensing-visualize-modalities` 不复制 parser/main 实现且 help 可用。
- [ ] 2.5 运行 `conda run -n kd_mm_beam kd-sensing-visualize-modalities --help` 和 `conda run -n kd_mm_beam kd-sensing-export-viewer-manifest --help` 验证入口。

## 3. 训练运行时状态与日志拆分

- [ ] 3.1 提取训练运行状态数据结构，集中保存 start epoch、best loss/top1、early stopping state、registry checkpoint、history 和 checkpoint loads。
- [ ] 3.2 提取 `EpochMetricsRecorder` 或等价 helper，负责 running metrics、history append、epoch_log 组装和 objective-specific 字段写入。
- [ ] 3.3 将 `training_outputs.npz` payload 构造迁入 history/artifact helper，并保持现有字段名和 inactive optional metric 的 NaN/null 语义。
- [ ] 3.4 将 TensorBoard startup、objective scalar、CRAF/MARF scalar 和 legacy accuracy tag 写入迁入专用 logger helper。
- [ ] 3.5 运行 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py tests/test_prediction_objectives.py -q` 验证训练日志和 objective 字段兼容。

## 4. checkpoint 与 artifact 写出拆分

- [ ] 4.1 提取 checkpoint manager，负责 `best.pth`、`best_top1.pth`、`last.pth`、sidecar 和 checkpoint registry archive。
- [ ] 4.2 保留 early stopping primary metric 与历史 Top-1 辅助 checkpoint 的选择语义，并保留历史 checkpoint fallback 恢复逻辑。
- [ ] 4.3 提取 artifact writer，负责 `resolved_config.yaml`、`final_config.yaml`、`train_log.json`、`teacher_metrics.json`、训练曲线和 debug artifacts。
- [ ] 4.4 更新相关 tests，验证 checkpoint sidecar、registry metadata、final config runtime metadata 和 resume 行为兼容。
- [ ] 4.5 运行 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py -q` 验证 checkpoint/artifact 路径。

## 5. BatchStepRunner 迁移

- [ ] 5.1 提取 batch step runner，封装 batch preparation、label/auxiliary target 准备、extension controls、student forward 和 base KD/no-KD loss。
- [ ] 5.2 将 `compute_prediction_loss`、extension `after_forward`、backward、grad clip、AMP scaler 和 optimizer step 编排接入 batch step runner。
- [ ] 5.3 确保 CRAF、MARF、G2D extension 的 hook 调用顺序和 diagnostics key 保持不变。
- [ ] 5.4 简化 `trainer.py` 的 batch 主循环，使其只调用 batch step runner 并把返回结果交给 recorder、health tracker 和 progress logger。
- [ ] 5.5 运行 `conda run -n kd_mm_beam pytest tests/test_g2d_distiller.py tests/test_g2d_loss.py tests/test_craf_fusion.py tests/test_marf_training.py -q` 验证方法扩展行为。

## 6. 配置加载 pipeline 拆分

- [ ] 6.1 提取 config source helper，负责实体 YAML、virtual canonical 配置和缺失配置错误，不改变 `load_config` 入口。
- [ ] 6.2 提取 config normalization helper，迁移 objective defaults、fusion modality selection、CSI hardening alias、LiDAR normalization、DeepSense scene、image profile 和 snapshot runtime requirements。
- [ ] 6.3 提取 config validation helper，迁移 cache policy、prediction objective、Raymobtime、image/radar profile 和 multitask validation。
- [ ] 6.4 提取 migration guard helper，迁移 removed image motion profile/cache/encoder 和其它历史拒绝规则。
- [ ] 6.5 将 Raymobtime s008 future/history/transition 禁用规则迁入 dataset rule helper 或等价 validation helper。
- [ ] 6.6 保持 `config/io.py` 只协调 source、overrides、normalization 和 validation，不直接维护业务规则表。
- [ ] 6.7 运行 `conda run -n kd_mm_beam pytest tests/test_student_configs.py tests/test_snapshot_next_frame_baselines.py tests/test_raymobtime_s008_selection.py -q` 验证配置解析兼容。

## 7. 架构边界与文档

- [ ] 7.1 扩展 `tests/test_architecture_boundaries.py`，检查训练方法逻辑不重新堆入 `trainer.py`，配置业务规则不重新堆入 `config/io.py`。
- [ ] 7.2 更新 README、扩展指南或工具文档中与训练编排、配置 pipeline、CLI 推荐入口相关的说明。
- [ ] 7.3 确认新增 helper 不扩大轻量导入边界，尤其 `kd_sensing.config`、`kd_sensing.registries`、`kd_sensing.engine.model_output` 仍不触发重依赖。
- [ ] 7.4 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` 验证架构边界。

## 8. 验证与收尾

- [ ] 8.1 运行 `openspec validate refactor-training-config-orchestration --strict` 并修复 OpenSpec 问题。
- [ ] 8.2 运行 `openspec status --change refactor-training-config-orchestration` 确认 proposal、design、specs 和 tasks 状态。
- [ ] 8.3 运行 focused regression：`conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py tests/test_training_io_workflow.py tests/test_prediction_objectives.py tests/test_raymobtime_s008_selection.py -q`。
- [ ] 8.4 运行 CLI smoke：`conda run -n kd_mm_beam kd-sensing-train --help`、`kd-sensing-evaluate --help`、`kd-sensing-preprocess --help`、`kd-sensing-export-viewer-manifest --help` 和 `kd-sensing-visualize-modalities --help`。
- [ ] 8.5 最终运行 `conda run -n kd_mm_beam pytest -q` 作为全量回归验收。
- [ ] 8.6 确认未将 `outputs/`、`logs/`、cache、checkpoint、真实数据或本地临时产物纳入源码变更。
