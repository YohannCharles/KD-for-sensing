## Context

旧 CSI hardening full sweep 中，A1/B/C/D 多数变体停在 `accuracy_val≈0.14`、`val_adba≈0.50`。后续 debug 矩阵已经证明 A0 original、A0 generated clone、pilot disabled、C1 view gate warmup only、C2 no internal GRU only 都能正常学习，因此配置生成、训练连接、view warmup 和 no-internal-GRU 路径本身不是主因。

真正异常来自旧 full sweep 的 pilot 配置：多个非 A0 变体继承了 `csi_estimation.mode: physical` 和 `noise_var: 0.01`。当前 CSI encoder 会先用训练集 RMS 归一化，再调用 pilot estimator。debug 首 batch 显示归一化后的 CSI 功率约为 `7.8e-9`，而 `0.01 / 16 = 6.25e-4` 的 complex 噪声方差会带来约 `8e4` 的 noise/signal power ratio。这不是 mild estimation noise，而是 destructive degradation。

当前需要把实验语义拉回控制变量矩阵：

```
          A0 clean baseline
                 │
       ┌─────────┴─────────┐
       │                   │
  pilot-only A1       hardening/encoder-only B/C/D
       │                   │
  相对 SNR 标定        显式 pilot mode: none
       │                   │
       └─────────┬─────────┘
                 ▼
          debug parity gate
                 ▼
          rerun and analyze
```

## Goals / Non-Goals

**Goals:**

- 修正 CSI pilot estimation 的 mild 噪声配置语义，避免在归一化后使用未校准的绝对 `noise_var`。
- 让 A1 pilot-only 实验使用相对信号功率的 estimation-SNR 模式，并记录 `noise_power/signal_power`。
- 让 B/C/D 单变量配置显式关闭 pilot noise，确保它们只测试 hardening 或 encoder 改动。
- 在分析脚本中加入有效性 gate：A0 parity、pilot 噪声量级、C1/C2 健康状态未通过时，停止候选排序和结论解释。
- 将旧 full sweep 输出标记为无效或待排查，避免继续被论文结论或候选选择使用。

**Non-Goals:**

- 不重新设计 CSI encoder 主体结构。
- 不改变 destructive degradation 作为 negative control 的存在价值。
- 不引入新的外部日志系统或实验跟踪依赖。
- 不在本 change 中解释 GPS+CSI/G2D 多模态收益；应先完成 CSI-only 矩阵修复和重跑。

## Decisions

1. A1 使用 estimation-SNR，不再使用未校准的 `physical noise_var` 表示 mild pilot。

   `est_snr` 直接以当前 estimator 输入张量的信号功率计算噪声方差，和归一化顺序无关。建议 A1 使用固定或训练采样的 25 到 35 dB 区间；对应 noise/signal power ratio 约为 `0.0032` 到 `0.00032`。这符合“上限不明显掉、收敛变慢”的目标。

   备选方案是继续使用 `physical noise_var` 并人工调小绝对值。该方式和数据 RMS、导出尺度、归一化位置强绑定，迁移到其它 scene 或 dataset 时容易再次失真。

2. 保留 `physical` 模式，但把它定位为已知输入尺度下的绝对噪声方差。

   物理模式本身没有错；错在把未归一化物理量直接放到归一化后的 estimator 输入上，还把它称为 mild。实现上应继续支持 `sigma_e2 = noise_var / (pilot_power * pilot_len)`，但 diagnostics 和分析 gate 必须暴露实际 `noise_power/signal_power`，并在 mild run 中检查其是否符合目标区间。

   备选方案是改变 `physical` 模式公式。该方式会破坏已有规格和可能依赖该公式的测试，不如通过配置语义和 gate 解决。

3. B/C/D 配置全部显式设置 `csi_estimation.mode: none`。

   这些组的变量应分别是 hardening 和 encoder 结构。即使默认值当前是 no-noise，也应在矩阵配置里显式写出，避免未来默认变化或模板继承再次引入 pilot 噪声。

   备选方案是依赖默认配置。旧 sweep 已经证明默认/继承路径容易造成隐性变量，不适合控制变量实验。

4. 分析脚本先判定 run validity，再做候选排序。

   `summary.csv` 和 `ranked_candidates.csv` 应包含 gate 状态，例如 `pilot_noise_scale_valid`、`full_sweep_status`、`invalid_reason`、`a0_parity_status`。当旧结果没有 debug diagnostics 或检测到 pilot 噪声失真时，输出应明确标记 invalid，而不是继续计算 `is_destructive` 或 `is_slow_high_ceiling` 作为主结论。

   备选方案是只在 README 中人工说明旧结果无效。该方式不能防止脚本或后续自动流程继续误用旧 CSV。

5. 修复后先做短跑 gate，再恢复长跑。

   最低顺序是：A0 original/clone、A1 est-SNR、C1/C2、B5/B6。只有 A0 clone parity 和 C1/C2 健康通过，才运行完整 A/B/C/D 50+ epoch 矩阵；只有 CSI-only 矩阵找到候选后，再进入 GPS+CSI 或 G2D validation。

## Risks / Trade-offs

- [Risk] estimation-SNR 的 dB 区间仍可能太弱或太强。→ Mitigation：先用 25/30/35 dB 小网格短跑，按 `noise_power/signal_power` 和 E90 曲线选择主 sweep 区间。
- [Risk] 旧 full sweep 结果被误删或无法追溯。→ Mitigation：不删除旧产物，只在分析输出和文档中标记 invalid，并保留原路径。
- [Risk] 物理模式仍被未来配置误用于 mild。→ Mitigation：为 mild run 增加 noise ratio gate，并在配置测试中检查 A1/B/C/D 的 estimator 模式。
- [Risk] debug gate 增加重跑耗时。→ Mitigation：gate 只跑 10 到 20 epoch，且只覆盖少量基准 run；通过后再投入长跑资源。

## Migration Plan

1. 修改 hardening matrix 配置或生成器：A1 改为 estimation-SNR；B/C/D 显式 `mode: none`；A2 保持 destructive negative control 但标识为 destructive。
2. 在 CSI estimator diagnostics 中确保 `sigma_e2`、`h_power_mean`、`noise_power_mean`、`noise_power_signal_ratio` 对所有 pilot-noise run 可用。
3. 在分析脚本中增加 validity gate 和旧结果 invalid 标记。
4. 扩展配置和 estimator 测试，覆盖 A1 噪声比例、B/C/D pilot 关闭、旧 physical 噪声失真检测。
5. 用 `conda run -n kd_mm_beam` 运行相关单元测试和短跑 gate。
6. 重跑 CSI-only A/B/C/D sweep，生成新的 summary 和 candidate ranking。

Rollback：保留旧配置文件备份或 Git diff 即可回退；分析脚本的 invalid 标记不会删除旧实验产物。

## Open Questions

- A1 主 sweep 的 SNR 区间最终用固定 30 dB，还是训练 25 到 35 dB 采样、验证固定 30 dB？首版建议采用训练采样并在 diagnostics 记录 sampled SNR。
- full sweep 长跑 epoch 是否继续使用旧 100 epoch/early stopping，还是先固定 50 epoch 做候选筛选？首版建议先固定 50 epoch，确认候选后再对少量配置延长训练。
