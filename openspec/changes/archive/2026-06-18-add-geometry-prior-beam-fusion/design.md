## Context

`add-predictive-gps-query-advantage` 已完成一轮 strict 新实验。结果显示 Predictive GPS-query++ 在 clean/P0、canonical P0-P5 和 GPS-query advantage slice 上都明显低于 `Image ResNet+GPS` 与 `JEPA GPS-query k=4`。该失败不是千分位波动，而是 clean test DBA 从 baseline 的约 0.88 降到 0.53，说明新 latent query/predictor/gate 路线破坏了主表征或训练分布。

外部文献给出的方向也更偏向几何对齐和训练稳定性：Vision-Position beam prediction 证明视觉和位置联合有效；position-aided 研究强调真实 GPS 噪声和评价指标边界；Multimodal Transformer beam prediction 使用 feature-level fusion、GPS angle calibration、focal loss、EMA 和数据增强；最新 BEV-Fusion 论文认为 1D pooled latent fusion 会丢几何结构，在 DeepSense S32-34 上通过 BEV 空间融合获得明显 DBA 增益。多模态训练论文还指出，多模态模型经常因容量、过拟合和模态泛化速度不同而输给单模态。

因此本方案不继续扩大 GPS-query latent path，而是把 GPS 变成可解释的 beam prior 和可靠性证据；image/fusion branch 仍承担主感知；loss 和 curriculum 优先保护 clean/P0 性能；所有 claim 必须通过 strict comparison gate。

## Goals / Non-Goals

**Goals:**

- 提供 opt-in Geometry-Prior Beam Fusion：从 GPS 几何特征生成 beam prior logits/distribution，再与 image/fusion logits 在 logit 层融合。
- 提供 DBA-aware supervised loss 和 beam topology smoothing，使训练目标更贴合 beam codebook 邻近关系，同时保留 hard-label validation/evaluation。
- 提供 teacher-guided stabilization，优先使用当前 strong `Image ResNet+GPS` strict checkpoint 约束 candidate 不偏离 clean 能力。
- 提供 clean-first curriculum：先跑 clean/control ablation，再逐步混入 P1-P5 和 advantage slice；禁止只在单个 hard condition 训练后升级 claim。
- 提供可审计 diagnostics：GPS prior quality、prior-image agreement、branch uncertainty/weights、teacher agreement、per-condition margins 和 clean regression gate。
- 保持所有真实训练、评测、图表和 checkpoint 产物在 ignored `outputs/` 或 `logs/` 下。

**Non-Goals:**

- 不恢复旧 KD、HiST、Top8 selector、camera residual、GPS residual 或 retired research line 的入口、配置或兼容 wrapper。
- 不把 Predictive GPS-query++ 结果包装成成功；旧 change 的 failed 结论保留为 provenance。
- 不把 advantage slice 替代为主 claim；clean/P0 和 canonical P0-P5 仍是主门槛。
- 不要求一次实现完整 BEV-Fusion 论文复现；BEV-lite 或 geometry prior 是 component baseline，完整 BEV-Fusion 复现需独立 workflow/paper reproduction 路径。
- 不让 condition id、P/C/D condition 名称、split 字符串或 claim label 进入模型 gate/fusion 输入。

## Decisions

### Decision 1: GPS 作为 beam prior，不再作为隐式 latent query 主路径

GPS 分支输出 `geometry_prior_logits` 或归一化 `geometry_prior_distribution`，class 维与 beam label space 对齐。输入优先使用 GPS-Rel-Polar，也可支持 relative Cartesian、angle calibration 和速度/历史差分。该 prior 可单独评估 top-k/DBA/entropy，并与 image branch 的 logits 做 late/logit-level fusion。

备选方案是继续扩大 `gps_query_attention` 或 Predictive++ 的 residual latent。最近 strict run 已证明该方向 clean 性能风险很大，且难以解释 GPS 何时帮助或误导。

### Decision 2: logit-level reliability fusion 优先于 latent-level black-box fusion

每个 branch 输出 logits 和可选 uncertainty/evidence。融合层只组合 logits/probabilities 和可靠性摘要，例如 image observability、GPS delay/counterfactual mask、branch entropy、teacher agreement、prior-image disagreement。这样普通 baseline 可以忽略 reliability metadata，candidate 也能输出可解释 branch weights。

备选方案是继续在 `[B,T,D]` latent 里用 MLP gate。该方案虽然灵活，但本次失败显示它容易破坏主表征，而且 diagnostics 聚合不到位时很难定位原因。

### Decision 3: clean-first curriculum 是 claim 前置条件

训练流程拆成三层：

1. clean/control：geometry prior 单独、image branch 单独、logit fusion、teacher-guided 版本必须先在 P0/clean 上接近 baseline。
2. P-suite：在 clean 为主的 mixed curriculum 中加入 P1-P5。
3. advantage slice：只作为机制诊断和 robustness evidence，不替代 P-suite。

如果 clean/P0 DBA 相对 `Image ResNet+GPS` 下降超过配置阈值，claim gate 直接失败，即使 advantage slice 有提升。

### Decision 4: DBA-aware loss 作为 supervised beam smoothing，不作为旧 KD 回归

DBA-aware loss 可以包括 circular Gaussian soft label、beam topology label smoothing、distance-aware CE、class-balanced focal circular loss或 EMD-style beam loss。它们使用 hard target 或允许的 beam power/source distribution 构造，不通过 retired distillation runtime。日志命名使用 `loss/beam_*` 或 `loss/geometry_prior_*`，不得使用旧 `loss/distillation` 路线。

Teacher-guided stabilization 是可选训练稳定项，仅使用明确声明的 strong checkpoint logits/probabilities，metadata 记录 teacher provenance、temperature、weight 和 detach 状态。实现不得导入或恢复 `kd_sensing.distillation` 子包。

### Decision 5: 完整 BEV-Fusion 复现另走 workflow 路径，当前先做 BEV-lite/geometry component

BEV-Fusion 文献方向很有价值，但完整 camera-to-BEV、LiDAR/radar/GPS BEV 和 temporal transformer 复现会跨多个模态与 workflow。当前 change 先做低风险的 geometry prior 和 logit fusion；若后续要复现论文表格，则在 `src/kd_sensing/baselines/<family>/` 或包内 CLI 中走 workflow/paper reproduction，并单独 OpenSpec。

### Decision 6: diagnostics 是实现的一等验收条件

本 change 的 candidate 不只看最终 DBA。每次 strict run 必须输出：

- geometry prior standalone DBA/Top-K/entropy；
- prior-image agreement、prior-teacher agreement、prior-target distance；
- logit fusion branch weights 或 evidence；
- clean/P0、P0-P5、advantage per-condition margins；
- teacher-guidance weight、temperature 和 loss 曲线；
- condition id isolation 标记。

没有 diagnostics 的 run 不允许升级 claim，只能作为 smoke 或 development evidence。

## Risks / Trade-offs

- [Risk] GPS prior 过强，错误 GPS 或 async GPS 误导 logits。  
  -> Mitigation: prior branch 输出 uncertainty，GPS reliability 低时降低权重；P3/A1/A2/CxD 必须纳入 gate diagnostics。

- [Risk] teacher-guided stabilization 被误解为恢复旧 KD。  
  -> Mitigation: 不使用 retired KD 配置、子包或入口；metadata 标记为 opt-in teacher-guided stabilization；日志使用独立命名。

- [Risk] clean-first curriculum 牺牲 hard-condition robustness。  
  -> Mitigation: curriculum 配置支持逐步提高 difficulty ratio，但 clean/P0 regression gate 始终保留。

- [Risk] DBA-aware soft target 让 Top-1 变差但 DBA 变好。  
  -> Mitigation: report 同时列出 Top-1/Top-3/Top-5/DBA；claim gate 同时检查 DBA 和最低 Top-K sanity。

- [Risk] 完整 BEV 方向更强，但当前 BEV-lite 不够。  
  -> Mitigation: 当前先验证几何 prior 是否能稳定超过 baseline；若不足，再单独提出 BEV-Fusion reproduction change。

- [Risk] 额外 diagnostics 增加实现工作量。  
  -> Mitigation: 先实现 CSV/JSON 聚合，再补图；没有聚合字段时图表必须标记 unavailable，不生成伪解释。

## Migration Plan

1. 新增 component 和 loss 的 synthetic tests，确保 shape、metadata、condition id isolation、ordinary baseline ignore metadata 先通过。
2. 新增 geometry prior standalone config，跑 GPS prior 单独诊断，确认 GPS prior 在 S32-34 上有合理 upper/lower bound。
3. 新增 image logits + geometry prior logits fusion config，先 clean-only smoke，再 strict clean/P0 evaluation。
4. 新增 DBA-aware supervised loss 和 teacher-guided stabilization ablation，不一次打开所有开关。
5. 新增 clean-first mixed curriculum，跑 P0-P5 与 advantage slice。
6. 更新 result claim registry / mainline experiment docs，只在 strict gate 通过时声明优势；否则记录 failed 或 pending。

Rollback 策略：禁用 geometry prior fusion 配置即可回到现有 `Image ResNet+GPS`、`JEPA GPS-query k=4` 和普通 supervised fusion；新增组件不改变默认训练入口。

## Open Questions

- GPS angle calibration 应优先复用现有 preprocessing/scaler，还是先在 model config 中实现轻量几何特征转换？
- DBA-aware loss 的第一版选择 circular Gaussian soft CE、distance-aware CE 还是 EMD-style loss？
- teacher-guided stabilization 的 teacher 只用 Image ResNet+GPS，还是同时支持 JEPA GPS-query k=4 ensemble？
- clean regression gate 阈值采用 DBA 下降不超过 0.01、0.02，还是按 baseline seed 方差估计？
- 是否需要马上新增 BEV-lite camera spatial prior map，还是先只做 GPS prior logits？
