## 1. G2D 基础结构

- [x] 1.1 在 `src/kd_sensing/distillation/` 新增 `g2d.py`、`teacher_ensemble.py`、`g2d_smp.py`，并在现有 distillation import/registry 路径中注册 `distillation.type: g2d`
- [x] 1.2 定义 G2D 配置解析结构，支持 `mode`、teacher 列表、loss 权重、temperature、horizons、feature align、logit align、SMP 和 diagnostics 字段
- [x] 1.3 实现通用 shape 校验工具，确保 G2D student logits、teacher logits 和 labels 严格满足 `[B,num_pred,C]`、`[B,num_pred,C]`、`[B,num_pred]`
- [x] 1.4 为错误信息补充模态名、期望 horizon、实际 shape、checkpoint 来源等上下文

## 2. Teacher Ensemble

- [x] 2.1 实现 `TeacherEnsemble`，按 `image`、`radar`、`gps`、`lidar`、`mmwave` 构建单模态 teacher
- [x] 2.2 支持显式 checkpoint 路径和当前场景 best checkpoint registry 解析；解析失败时 fail fast
- [x] 2.3 加载 teacher checkpoint 时默认 `strict_load: true`，并把所有 teacher 设为 `eval()` 和 `requires_grad=False`
- [x] 2.4 实现 teacher ensemble forward，复用现有 batch preparer，为每个单模态 teacher 返回 `ModelOutput`
- [x] 2.5 增加 teacher logits horizon 校验，禁止 `[B,4,64]` teacher 输出被静默截断

## 3. G2D Loss 与特征提取

- [x] 3.1 实现 supervised CE，默认对 `t+1/t+2/t+3` 全部 horizon 计算 loss
- [x] 3.2 实现 logit KD，对每个 teacher 计算 temperature KL 并对模态求平均，确保 teacher logits detach
- [x] 3.3 实现 feature extraction helper，支持 `input_features`、`output_features`、`diagnostics["modality_features"]`、`token_features` 和 dict/tensor 格式
- [x] 3.4 实现 feature KD，支持 `last`/`mean` pooling、normalize、MSE，以及每模态 auto projection
- [x] 3.5 让 `G2DDistiller.compute()` 返回 total loss、loss breakdown、teacher confidence、ranking 和可聚合 diagnostics

## 4. Fusion Student 接入

- [x] 4.1 扩展 legacy `fusion_student` forward 输出，在不改变 logits 契约的前提下通过 diagnostics 暴露 per-modality branch features
- [x] 4.2 确保 CRAF/MARF/token transformer 模型的 `token_features`、`modalities` diagnostics 能被 G2D feature extractor 按模态拆分
- [x] 4.3 为 `adapt_model_output()` 或 G2D feature extractor 增加必要测试，避免破坏旧 tuple/tensor/dict 输出解析

## 5. Trainer 与 SMP

- [x] 5.1 修改 `src/kd_sensing/engine/trainer.py`，仅在 `distillation.type: g2d` 时构建 `TeacherEnsemble` 并走 G2D 训练 step
- [x] 5.2 保持 no-KD、logits KD、RKD、CRAF 和 MARF 现有训练路径不变
- [x] 5.3 实现 `SMPScheduler`，按 teacher confidence 三步平均值输出 weak-to-strong ranking 和 active modalities
- [x] 5.4 实现 `apply_smp_gradient_mask()`，支持 `model.encoders.<modality>` 和 legacy `fusion_student` 分支命名
- [x] 5.5 在 AMP 与非 AMP 两种路径中，于 backward 后、optimizer step 前应用 SMP mask，并保持 grad clip 顺序正确

## 6. Diagnostics 与 Metrics

- [x] 6.1 新增 `src/kd_sensing/diagnostics/g2d_diagnostics.py`，实现 batch 聚合和 epoch JSON 写出
- [x] 6.2 诊断 JSON 写入 `diagnostics/g2d_epoch_<epoch>.json`，包含 `num_pred`、`horizon_names`、teacher confidence、ranking、active modalities 和 loss breakdown
- [x] 6.3 student branch confidence 可用时，写出 student branch confidence 和 confidence ratio
- [x] 6.4 扩展 validator/metrics 输出 `val_top1_t1/t2/t3/avg`、`val_top3_t1/t2/t3/avg`、`val_top5_t1/t2/t3/avg`
- [x] 6.5 确认 metrics 不新增 `top1_h0`、`top1_future_avg`、`beam8_acc` 或其它 current beam 字段

## 7. 配置与结果汇总

- [x] 7.1 新增 `configs/fusion/image_radar_gps_lidar_mmwave_g2d_lite.yaml`
- [x] 7.2 新增 `configs/fusion/image_radar_gps_lidar_mmwave_g2d_global.yaml`
- [x] 7.3 新增 `configs/fusion/image_radar_gps_lidar_mmwave_g2d_horizon.yaml`
- [x] 7.4 确保三个配置都设置 `experiment.task: fusion`、五模态 `modalities`、`model.num_pred: 3`、`distillation.type: g2d`
- [x] 7.5 新增 `tools/analysis/collect_multimodal_imbalance_results.py`，读取 metrics、train log 和 G2D diagnostics 并输出 CSV 汇总
- [x] 7.6 更新 README 或实验文档，记录 G2D 三个入口、teacher checkpoint 要求和结果汇总命令

## 8. 测试

- [x] 8.1 新增 `tests/test_g2d_loss.py`，覆盖 loss scalar、权重关闭、detach、shape 错误和 `[B,4,64]` 拒绝
- [x] 8.2 新增 `tests/test_g2d_distiller.py`，覆盖 teacher ensemble 输出、teacher confidence 和 feature/logit KD 组合
- [x] 8.3 新增 `tests/test_g2d_smp.py`，覆盖 weak-to-strong 排序、active modality schedule 和 gradient mask
- [x] 8.4 新增 `tests/test_g2d_diagnostics.py`，覆盖 diagnostics JSON schema、horizon names、ranking 和 loss 字段
- [x] 8.5 新增或扩展配置 smoke tests，确认三个 G2D YAML 可加载并解析为五模态 G2D 配置

## 9. 验证

- [x] 9.1 运行 `conda run -n kd_mm_beam pytest -q tests/test_g2d_loss.py tests/test_g2d_distiller.py tests/test_g2d_smp.py tests/test_g2d_diagnostics.py`
- [x] 9.2 运行 `conda run -n kd_mm_beam pytest -q`
- [x] 9.3 使用 Scene9 或 synthetic/小比例数据运行 `conda run -n kd_mm_beam python scripts/train.py --config configs/fusion/image_radar_gps_lidar_mmwave_g2d_lite.yaml -o training.epochs=1`
- [x] 9.4 运行 `conda run -n kd_mm_beam python scripts/train.py --config configs/fusion/image_radar_gps_lidar_mmwave_g2d_global.yaml -o training.epochs=1`，确认 active modalities 记录正确
- [x] 9.5 运行结果汇总脚本 `conda run -n kd_mm_beam python tools/analysis/collect_multimodal_imbalance_results.py`，确认输出 CSV 包含 top-k、teacher confidence、ranking 和 final active modalities 字段
