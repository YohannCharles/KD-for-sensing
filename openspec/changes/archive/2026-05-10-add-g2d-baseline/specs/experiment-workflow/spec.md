## ADDED Requirements

### Requirement: G2D training workflow
训练入口 MUST 支持通过配置启动 G2D 训练。G2D 训练 MUST 使用 fusion student 作为可训练主模型，MUST 使用多个 frozen 单模态 teacher，MUST 保存常规训练产物和 G2D diagnostics。

#### Scenario: 启动 G2D-lite 训练
- **WHEN** 用户运行 `conda run -n kd_mm_beam python scripts/train.py --config configs/fusion/image_radar_gps_lidar_mmwave_g2d_lite.yaml`
- **THEN** 系统 MUST 构建 fusion student
- **AND** 系统 MUST 构建并冻结配置中的单模态 teacher ensemble
- **AND** 系统 MUST 使用 supervised CE、feature KD 和 logit KD 完成训练 step

#### Scenario: 启动 G2D-global 训练
- **WHEN** 用户运行 `conda run -n kd_mm_beam python scripts/train.py --config configs/fusion/image_radar_gps_lidar_mmwave_g2d_global.yaml`
- **THEN** 系统 MUST 执行 G2D 训练 step
- **AND** 系统 MUST 在 optimizer step 前应用 SMP 梯度屏蔽
- **AND** 训练日志或 diagnostics MUST 记录当前 active modalities

#### Scenario: 启动 G2D-horizon 诊断训练
- **WHEN** 用户运行 `conda run -n kd_mm_beam python scripts/train.py --config configs/fusion/image_radar_gps_lidar_mmwave_g2d_horizon.yaml`
- **THEN** 系统 MUST 运行 G2D 训练
- **AND** 每个 epoch diagnostics MUST 记录 `t+1`、`t+2` 和 `t+3` 的 modality ranking

### Requirement: Future horizon flat metrics
验证和评估输出 MUST 在现有 nested top-k 数组之外，增加 future horizon 扁平指标字段。字段 MUST 使用 `t1/t2/t3/avg` 命名，并 MUST 不输出历史 current beam 或 h0 指标。

#### Scenario: 保存三步 Top-K 扁平字段
- **WHEN** 验证阶段产出 logits `[B,3,64]` 和 labels `[B,3]`
- **THEN** `metrics.json` MUST 包含 `val_top1_t1`、`val_top1_t2`、`val_top1_t3` 和 `val_top1_avg`
- **AND** `metrics.json` MUST 包含 `val_top3_avg` 和 `val_top5_avg`
- **AND** 这些 avg 字段 MUST 对有效 future horizon 求平均

#### Scenario: 不输出旧 h0 指标
- **WHEN** G2D 或普通 future-only 评估写出 metrics
- **THEN** metrics MUST 不包含 `top1_h0`
- **AND** metrics MUST 不包含 `top1_future_avg`
- **AND** metrics MUST 不包含 `beam8_acc`

### Requirement: G2D validation commands
G2D 实现 MUST 提供定向测试和 smoke training 验证命令，并且所有 Python 命令 MUST 使用 `conda run -n kd_mm_beam`。

#### Scenario: 运行 G2D 定向测试
- **WHEN** 开发者验证 G2D 实现
- **THEN** 推荐测试命令 MUST 为 `conda run -n kd_mm_beam pytest -q tests/test_g2d_loss.py tests/test_g2d_distiller.py tests/test_g2d_smp.py tests/test_g2d_diagnostics.py`
- **AND** 测试 MUST 覆盖 loss shape、teacher confidence、SMP scheduler、gradient mask 和 diagnostics schema

#### Scenario: 运行 G2D smoke training
- **WHEN** 开发者完成 G2D 实现
- **THEN** 开发者 MUST 能使用 `conda run -n kd_mm_beam python scripts/train.py --config configs/fusion/image_radar_gps_lidar_mmwave_g2d_lite.yaml -o training.epochs=1`
- **AND** 该 smoke run MUST 完成 forward、loss、backward、optimizer step、validation 和 diagnostics 保存
