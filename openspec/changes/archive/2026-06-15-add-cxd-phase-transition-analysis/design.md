## Context

当前 `kd-sensing-jepa-gps-shortcut-benchmark` 已能解析 Scenario C、Scenario D 与 CxD joint suite，并写出 `results/scenario_d_image_observability.csv`、`results/heatmap_cx_dy.npy` 以及基础 PNG 图表。现有 smoke 主要验证 manifest、聚合 schema 和 artifact 边界；`modality_dominance_ratio` 等字段仍偏向模型组启发式，不能作为论文中“JEPA 改变模态主导关系”的强解释证据。

本 change 要把 Scenario D CxD 输出升级为只读分析能力：复用现有 benchmark runner、difficulty pipeline、模型组可比性检查和 ignored output 边界，在本地真实 checkpoint 可用时输出 phase diagram、dominance、crossover 和 failure-mode decomposition；在 synthetic 或缺诊断模型上只输出可审计的 skipped/unavailable 状态。

## Goals / Non-Goals

**Goals:**

- 为每个模型、每个 `(Cx, Dy)` 输出统一 long-form metrics、3D heatmap matrix、robustness ratio、worst-case、RSI 和 clean-relative drop。
- 从真实诊断源计算 GPS/image/JEPA latent contribution score，包括 gradient norm ratio、attention/fusion dominance、JEPA latent variance 或 manifest 声明的等价诊断。
- 检测 CNN+GPS 与 Image-JEPA/Image-JEPA+GPS/query_pool 的 crossing region，并输出 crossing boundary、regime label 和 query_pool 相对 biased 的 shift summary。
- 将 C4+D7 及 C/D 单轴极端退化拆解为 GPS fail dominant、image fail dominant、both fail 或 unavailable。
- 保持 benchmark 输出只写入 `outputs/analysis/...` 或用户显式 output dir，不提交真实 CSV、NPY、PNG、checkpoint、cache 或 report。
- 提供 synthetic focused tests 验证 schema、聚合、绘图降级、diagnostic unavailable 和 no label shift guard。

**Non-Goals:**

- 不新增训练流程或模型主线，不改变 JEPA/query_pool/supervised baseline 的训练配置。
- 不恢复 KD、HiST/Hist、Top8 selector、GPS residual、camera residual 或 root-level legacy script。
- 不把 attention、gradient 或 latent variance 单独声明为因果证明；它们只作为解释性诊断证据。
- 不要求所有模型都支持 attention/gradient diagnostics；不支持的模型必须显式记录 skipped reason。
- 不提交真实 benchmark 结果、论文图或 checkpoint。

## Decisions

### 1. 复用 benchmark runner，新增分析 helpers

实现放在 `src/kd_sensing/diagnostics/jepa_gps_shortcut_benchmark.py` 的窄 helper 中，优先拆分为纯函数：`aggregate_cxd_phase_diagram`、`compute_modality_dominance`、`detect_cnn_jepa_crossing`、`decompose_cxd_failure_modes` 和 `write_cxd_phase_artifacts`。CLI 仍使用 `kd-sensing-jepa-gps-shortcut-benchmark`。

理由：Scenario C/D manifest、difficulty provenance、model comparability 和 output registry 已在该 runner 中存在，继续复用可以避免新入口和重复 schema。替代方案是新增独立 visual-analysis CLI，但会把同一 CxD matrix 的产物分裂到两个工作流中；本 change 只在后续可选让 `jepa_visual_analysis` 只读消费 runner manifest。

### 2. dominance 采用分层真实诊断源

每个 `(model, Cx, Dy)` 的 dominance 行包含 `gps_contribution_score`、`image_contribution_score`、`jepa_latent_contribution_score`、`diagnostic_source` 和 `status`。计算优先级为：

1. 显式 per-condition diagnostic artifact，例如 gradient norm、attention/fusion weights、latent variance CSV/NPZ。
2. 模型 forward diagnostics 或 batch-level metadata 中已暴露的 attention/reliability/latent fields。
3. manifest 声明的 external diagnostic summary。
4. unavailable/skipped，而不是使用模型组启发式替代。

理由：论文解释需要可追溯证据，启发式 dominance 会混淆“模型类别假设”和“观测到的行为”。替代方案是保留当前 `modality_dominance_ratio` 简化字段；它适合 smoke，但不适合作为正式 claim。

### 3. crossing detection 基于模型组配对和严格可比字段

crossing 检测只在同一 seed、split、label space、metric profile、sample_count 和 difficulty digest 下执行。CNN 侧默认包含 `cnn_gps`、`image_ae_gps`、`resnet_image_gps`；JEPA 侧默认包含 `image_jepa_only`、`image_jepa_gps`、`jepa_mean_pool`、`jepa_gps_query_pool`。manifest 可声明 paired model names，严格模式下缺配对则记录 blocked/unavailable。

理由：crossing point 是论文核心图，必须避免把不同 split 或不同 metric 的模型混成边界。替代方案是取所有 CNN 与 JEPA 模型的全局 best，但会掩盖 query_pool vs biased 的 paired comparison。

### 4. artifact 分为机器可读结果和论文图

runner 输出至少包括：

- `results/cxd_phase_diagram.csv`
- `results/cxd_phase_heatmap.npy`
- `results/modality_dominance.csv`
- `results/crossing_region_Cx_Dy.json`
- `results/failure_mode_decomposition.csv`
- `plots/cxd_accuracy_heatmap.png`
- `plots/cnn_jepa_crossover_curve.png`
- `plots/modality_dominance_heatmap.png`
- `plots/robustness_surface.png`

PNG/SVG/PDF 图表是可选产物；CSV/JSON/NPY 是核心产物。matplotlib 或可视化依赖不可用时，必须保留机器可读文件并在 manifest 中记录 warning。

理由：论文图生成不应阻塞结果审计。替代方案是只输出 PNG，但不利于复核和后续 paper packaging。

### 5. failure mode decomposition 使用相对下降与单轴参考

failure mode 以 clean `(C0,D0)`、GPS-only axis `(Cx,D0)`、image-only axis `(C0,Dy)` 和 joint `(Cx,Dy)` 的主指标为输入。若 joint drop 主要由 GPS axis 解释，标记为 `gps_fail_dominant`；主要由 image axis 解释，标记为 `image_fail_dominant`；两者均超过阈值或存在超加性下降，标记为 `both_fail`；缺少参考行时标记为 `unavailable`。

理由：这能把 C4+D7 worst-case 拆成 reviewer 可理解的失败机制。替代方案是只看 worst-case accuracy，会丢失 C/D 两个轴的归因。

## Risks / Trade-offs

- [Risk] 真实模型不暴露 gradient/attention/latent diagnostics → Mitigation: dominance row 标记 `unavailable`，不阻塞 metrics/heatmap/crossing 输出；tests 覆盖缺失诊断降级。
- [Risk] gradient probe 增加评估成本或需要保留输入梯度 → Mitigation: 通过 manifest 显式 opt-in，默认只读已有 diagnostic artifacts；大规模真实 run 可先跑 metrics，再单独补 dominance probe。
- [Risk] crossing boundary 被不同 split/metric 污染 → Mitigation: 严格复用 comparability metadata；不可比较模型只进入隔离组，不写入 crossing summary。
- [Risk] synthetic smoke 输出被误当论文结果 → Mitigation: manifest、docs 和 claim registry 明确 `mock/smoke`，真实数值只记录 ignored output 路径和 caveat。
- [Risk] 新 artifact 文件名与现有 Scenario D 输出重复或漂移 → Mitigation: 保留已有 `scenario_d_image_observability.csv` 与 `heatmap_cx_dy.npy`，新增 CxD phase artifact；runner manifest 同步列出全部 output files。

## Migration Plan

1. 在现有 Scenario D smoke manifest 上增加 analysis/dominance/crossing 配置字段，并保持旧字段兼容。
2. 实现纯函数聚合和 artifact writer，保证旧 `results/scenario_d_image_observability.csv`、`results/heatmap_cx_dy.npy` 和 `plots/*.png` 继续存在。
3. 增加 focused tests，先覆盖 synthetic rows，再覆盖 diagnostics unavailable、paired crossing 和 failure-mode decomposition。
4. 同步 docs 四层账本，将该能力标记为 smoke/evaluation-only；真实 checkpoint matrix 仍需用户本地替换权重后运行。

Rollback 策略是保留旧 Scenario D artifact，禁用新 `analysis.phase_transition.enabled` 或跳过新 writer；旧 benchmark smoke 和 visual-analysis ingestion 不应受影响。

## Open Questions

- 真实 gradient norm probe 是否在第一版就纳入 runner，还是先读取外部 diagnostic CSV/NPZ 后再增量接入模型 forward hook？
- query_pool vs biased 的 crossing shift 默认按 `gps_biased`/`gps_query_pool` group 自动配对，还是要求 manifest 显式声明 paired model names？
- 正式论文图默认 metric 用 DBA 还是 Top-3 DBA/primary metric，需要在真实 run manifest 中固定。
