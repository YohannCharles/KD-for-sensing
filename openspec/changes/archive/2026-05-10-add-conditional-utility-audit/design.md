## Context

当前代码已经具备 Phase 1 的一部分基础：

- `src/kd_sensing/engine/validator.py` 支持 `evaluation.modality_subsets.enabled`，并能通过 MARF 的 `force_modality_mask` 做 aggregate subset evaluation。
- MARF 模型已经声明 `supports_force_modality_mask`，且现有配置 `configs/fusion/scene32_marf.yaml` 默认启用五模态与 subset metrics。
- `src/kd_sensing/evaluation/metrics.py` 已提供 Top-K 与 DBA 聚合指标，但没有逐样本 DBA contribution 或 CE dump。
- `src/kd_sensing/distillation/teacher_ensemble.py` 已能构建冻结单模态 teacher，并规范化 teacher logits 到 `[B, H, C]`。
- `DeepSense6GDataset.__getitem__()` 当前只返回模态张量、`input_beam` 和 `target_beam`，没有稳定 `sample_id`、`seq_id` 或 `frame_idx`。

因此，本变更应以“独立诊断分析”为主：补齐 subset registry、逐样本记录、teacher oracle、通信状态分桶和图表，不把 Phase 2 的 MARF-Comm、router 输入、loss、encoder 解冻等策略混进来。

## Goals / Non-Goals

**Goals:**

- 用一个已训练 MARF checkpoint 运行 Scene32 Conditional Utility Audit，输出 aggregate、per-sample、oracle、bucket、teacher complementarity 和 figures。
- 统一 conditional audit subset 定义，并让现有 validator 和新分析入口复用同一 registry。
- 对 `t+1`、`t+2`、`t+3` 分别记录 Top1、Top3、Top5、DBA、CE、gt probability 和 top-k prediction。
- 从现有 teacher registry 严格加载单模态 teacher，判断弱模态是否包含 strong path 没有利用的补充信息。
- 在 summary 中给出初步 diagnosis 和 Phase 2 建议，但不自动修改模型或训练配置。

**Non-Goals:**

- 不修改 MARF 主结构、router、loss、subset training 策略或 encoder 冻结策略。
- 不新增通信状态特征作为模型输入。
- 不重新训练任何模型，也不把 teacher oracle 结果反向写入训练过程。
- 不改变现有训练、评估和 metrics JSON 的默认行为；conditional audit 只在显式配置或独立脚本下运行。

## Decisions

1. 新增独立 audit runner，而不是把所有逻辑塞进 `validate()`。

   `validate()` 保持轻量聚合评估，新增 `tools/analysis/run_conditional_utility_audit.py` 负责加载配置、checkpoint、dataloader、MARF 模型、teacher ensemble 和输出目录。validator 可以复用 subset registry 以减少硬编码，但逐样本 dump 与 teacher forward 由 audit runner 触发，避免普通验证变慢。

2. subset 定义放在 `src/kd_sensing/evaluation/subset_specs.py`。

   该模块暴露 Scene32 conditional audit subset registry，并用 `normalize_modalities()` 强制遵守 `image -> radar -> gps -> lidar -> mmwave` 的中心契约。现有 validator 的 `_modality_subset_definitions()` 应收敛到该 registry 或兼容包装，避免 `strong_only`、`weak_only`、`single_best_mmwave` 在多个地方分叉。

3. 逐样本数据先采用 DataFrame writer，并支持 parquet 优先、`csv.gz` fallback。

   如果环境中存在 `pyarrow` 或 pandas parquet engine，则写 parquet；否则写 `csv.gz`。summary metadata 必须记录实际格式。这样不强制引入新依赖，也能处理验证集行数扩大后的文件体积。

4. per-sample 指标复用现有指标语义，但新增局部 helper。

   聚合 DBA 继续来自 `calculate_dba_score()`，逐样本 DBA contribution 使用同样的 Top-3 和 `evaluation.dba_delta` 公式实现到 diagnostics helper 中，并用测试保证与 aggregate DBA 平均一致。CE 使用 `-log(gt_prob + eps)`，忽略 label `-100`。

5. teacher complementarity 复用 G2D teacher ensemble 语义。

   audit 使用 teacher registry 中的 checkpoint 构建 `image/radar/gps/lidar/mmwave` teacher，严格加载权重、`eval()`、`requires_grad=False`、`torch.no_grad()`，并要求 logits 规范化后为 `[B, 3, 64]`。这比重新实现 teacher loading 更贴近现有 G2D 代码。

6. metadata 扩展保持最小且只增加字段。

   Dataset 可新增 `return_metadata` 或等价配置，在 audit 配置中启用后返回 `dataset_index`、`sample_id` 和可用路径派生字段。普通训练配置默认关闭，避免影响既有 batch shape 与性能。若无法可靠解析 `seq_id/frame_idx`，必须保留 `dataset_index` 和 beam/mmWave/GPS path 片段用于追踪，而不能伪造含义不明的字段。

7. 通信状态特征只用于分桶分析。

   `communication_state_features.py` 从当前 batch 的 mmWave、GPS relative polar、`input_beam` 和 `target_beam` 计算 feature。mmWave entropy/margin 基于 power vector softmax 或非负归一化实现；GPS bearing 用 `atan2(sin_theta, cos_theta)`；beam transition 优先使用最后一个历史 beam 与 future label，不可用时降级到 future-to-future transition。分桶阈值按验证集分位数统一计算。

8. diagnosis 规则配置化。

   `min_bucket_samples`、`conditional_delta_dba`、`teacher_rescue_rate`、`oracle_gain_dba` 等阈值来自 audit 配置。summary 输出 diagnosis 时必须同时写入触发依据，便于人工判断规则是否过于激进。

## Risks / Trade-offs

- 逐样本 dump 文件较大 → 默认只在 audit runner 中生成，支持 parquet 和 `csv.gz`，并允许配置关闭 teacher dump 或 figures。
- teacher checkpoint 或 normalization artifact 缺失 → runner 在开始时 fail fast，错误信息包含 modality、registry path、checkpoint path 和 strict_load 设置。
- dataset metadata 不完整 → 使用稳定 `dataset_index` 与路径派生 fallback，并在 summary metadata 中声明 `seq_id/frame_idx` 是否可用。
- mmWave 归一化后的数值不再是原始功率 → bucket 特征先基于模型实际输入，summary 标记 `mmwave_source=normalized_input`；后续如需原始功率可单独扩展 dataset 返回原始特征。
- 逐样本 DBA helper 与 aggregate DBA 可能漂移 → 增加 toy 测试，验证逐样本 DBA 平均值与 `calculate_dba_score()` 一致。
- 把 validator 改大可能影响常规评估 → validator 只读取 subset registry；conditional audit 的重逻辑放在独立 diagnostics 模块和脚本中。

## Migration Plan

1. 新增 diagnostics 模块、subset registry、analysis config 和 runner，不改变现有配置默认行为。
2. 将 validator 内部 subset 定义迁移到 registry，并用现有 MARF subset 测试覆盖兼容名称。
3. 增加 dataset metadata opt-in 字段；普通训练配置不启用。
4. 增加单元测试和 dummy end-to-end audit。
5. 使用 `conda run -n kd_mm_beam pytest -q tests/test_subset_specs.py tests/test_conditional_utility_metrics.py tests/test_conditional_utility_oracle.py tests/test_communication_state_features.py` 做定向验证；最后运行 `conda run -n kd_mm_beam pytest -q`。

## Open Questions

- 当前 Scene32 CSV 是否包含足够字段可靠解析 `seq_id` 与 `frame_idx`。如果没有，首版只承诺稳定 `dataset_index/sample_id` 和路径追踪。
- 运行环境是否已有 parquet engine。没有时首版写 `csv.gz`，不强制安装 `pyarrow`。
- 实际 MARF checkpoint 文件名是 `best_top1.pt` 还是 `best_top1.pth`、`best.pth`。配置应允许覆盖，并复用现有 checkpoint resolution。
