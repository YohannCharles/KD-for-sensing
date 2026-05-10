## Why

当前项目已经切换为未来三步 beam 预测，标签与 logits 约定为 `[B, 3]` 和 `[B, 3, 64]`，但还缺少一个可与 CRAF/MARF 对照的通用多模态失衡 baseline。G2D 可以用多个单模态 teacher 同时指导 fusion student，并用 teacher confidence 诊断和缓解强模态压制弱模态的问题，适合补齐 G2D 对比实验文档中的实验矩阵。

## What Changes

- 新增 G2D baseline，覆盖 `g2d-lite`、`g2d-global`、`g2d-horizon` 三种运行模式。
- 新增多单模态 teacher ensemble 加载与前向，要求每个 teacher 输出严格匹配 `[B, 3, 64]`，checkpoint 缺失或 horizon 不匹配时直接报错。
- 新增 G2D loss：supervised CE、feature KD、logit KD，并支持按 horizon 选择训练范围，默认使用全部 `t+1/t+2/t+3`。
- 新增基于 teacher confidence 的 Sequential Modality Prioritization，`g2d-global` 使用三步平均 confidence 做弱到强排序并屏蔽 inactive modality encoder 梯度。
- 新增 G2D 诊断输出：teacher confidence、student branch confidence、confidence ratio、weak-to-strong ranking、active modalities、horizon-wise top-k 指标。
- 新增五模态 G2D 配置入口与多模态失衡结果汇总脚本。
- 保持 CRAF/MARF 现有语义不变；G2D 作为对照 baseline，不替代 MARF 主方法。

## Capabilities

### New Capabilities

- `g2d-multimodal-distillation`: 定义未来三步预测下的 G2D 多 teacher 蒸馏、SMP 调制、诊断和结果汇总能力。

### Modified Capabilities

- `configurable-multimodal-fusion`: 增加五模态 G2D fusion 配置入口，并要求解析后配置明确 `distillation.type: g2d` 及三种 G2D mode。
- `experiment-workflow`: 增加 G2D 训练、验证、诊断保存和汇总流程，所有 Python 命令继续通过 `conda run -n kd_mm_beam` 执行。

## Impact

- 影响训练路径：`src/kd_sensing/engine/trainer.py`、`src/kd_sensing/engine/batch.py`、`src/kd_sensing/engine/model_output.py`、`src/kd_sensing/engine/_builders_impl.py`。
- 影响蒸馏与损失：`src/kd_sensing/distillation/`，新增 G2D distiller、teacher ensemble、loss、SMP helper。
- 影响诊断和评估：`src/kd_sensing/diagnostics/`、`src/kd_sensing/evaluation/metrics.py`、`tools/analysis/`。
- 影响配置：`configs/fusion/image_radar_gps_lidar_mmwave_g2d_lite.yaml`、`configs/fusion/image_radar_gps_lidar_mmwave_g2d_global.yaml`、`configs/fusion/image_radar_gps_lidar_mmwave_g2d_horizon.yaml`。
- 影响测试：新增 G2D loss、teacher confidence、SMP、gradient mask、diagnostics 和配置 smoke tests，并保留既有 CRAF/MARF 回归测试。
