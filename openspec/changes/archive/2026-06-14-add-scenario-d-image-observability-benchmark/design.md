## Context

项目当前已经有统一 difficulty pipeline、Scenario C / GPS async preset、JEPA GPS shortcut benchmark、Image+GPS JEPA downstream、GPS-query pooling 和主线实验文档。Scenario D 不需要另起一套 dataset 或 runner；它应该作为现有 difficulty/benchmark 能力的延伸，用相同 batch contract、split、metric、seed 和 output boundary 验证“GPS 不可靠且视觉结构成为稳定线索”时 Image-JEPA 的结构预测优势。

本设计遵守当前项目边界：源码实现放在 `src/kd_sensing` 包内；所有项目 Python 命令使用 `conda run -n kd_mm_beam ...`；真实数据、checkpoint、cache、CSV/NPY/PNG 运行结果只写入 ignored 的 `outputs/`、`logs/` 或 manifest 指定目录；不新增退役 KD/HiST/residual/standalone Top8 路线，也不新增绕过包结构的仓库根入口。

## Goals / Non-Goals

**Goals:**

- 提供 Scenario D 图像可观测性坍塌 preset，覆盖 `D0_full_image` 到 `D7_joint_worst_case`。
- 复用 Scenario C 的 `C0_sync` 到 `C4_severe_async`，生成 `5 x 8` Cx-Dy robustness grid。
- 对 GPS-only、CNN+GPS、Image-AE+GPS、Image-JEPA only、Image-JEPA+GPS 执行严格可比较评估。
- 在 batch metadata 中显式建模 `image_valid_mask`、`image_observability_score`、image dropout/burst/dropout/corruption metadata、`gps_valid_mask` 和 `gps_delay_steps`。
- 新增 observability-aware fusion，使模型能用可靠性权重、uncertainty gating 和 JEPA temporal latent fallback 处理视觉/GPS 退化。
- 输出论文级表格和图，包括 Cx-Dy heatmap、worst-case、RSI、phase transition、CNN vs JEPA crossing point 和 modality dominance ratio。

**Non-Goals:**

- 不把 corruption 写入真实 `dataset/`，不改 split CSV，不移动 target label 或 beam power。
- 不提交真实 benchmark 结果、checkpoint、cache、plots 或 local reports。
- 不新增 root-level `datasets/` 包或旧式脚本入口；用户给出的 `ImageObservabilityTransform` 类名可在 `src/kd_sensing.data.difficulty` 下实现并导出。
- 不改 Stage 1 JEPA 预训练 checkpoint schema、target encoder EMA 或 JEPA latent prediction loss；本 change 只扩展 supervised downstream / benchmark runtime。
- 不保证某个真实训练结果一定出现 JEPA 优势；本 change 保证实验条件、模型输入和指标产物能检验该假设。

## Decisions

1. **Scenario D 作为 difficulty operator/preset，而不是独立 dataset transform 文件树。**

   实现落点为 `src/kd_sensing/data/difficulty/operators/image_observability.py` 或扩展现有 `operators/image.py`，并通过 `DIFFICULTY_OPERATORS` 注册。这样训练、评估和 benchmark 能复用同一 seed、metadata、target-preservation guard 和 replay manifest。备选方案是在仓库根新增 `datasets/image_observability.py`，但这会绕开当前 `src/kd_sensing` 包结构和统一 difficulty pipeline。

2. **保留用户语义中的 `ImageObservabilityTransform`，但让它成为包内 operator/helper。**

   类接口保留 `image_dropout_prob`、`image_burst_dropout_prob`、`max_burst_len`、`image_weather_severity`、`image_blur_prob`、`image_occlusion_prob`、`image_occlusion_ratio`、`image_lowlight_prob` 和 `seed`，同时补充 batch/operator wrapper 负责写入 `image_valid_mask`、`image_observability_score` 和 replay metadata。这样既保留研究描述中的可读接口，又不破坏 runtime contract。

3. **D-level preset 由 manifest/profile 标准化为 operator 参数。**

   `D0` 为 clean；`D1` 使用 weather severity；`D2` 使用 low-light probability；`D3` 使用 motion blur probability；`D4` 使用 occlusion probability/ratio；`D5` 使用 frame dropout；`D6` 使用 burst missing；`D7` 组合 `D4 + D6`，并在 Cx-Dy 矩阵中与 `C3/C4` 形成重点 worst-case 分析。这样可在 smoke、lowmem 和正式 manifest 中共享同一套 condition id。

4. **图像 corruption 和 image missing 分离。**

   weather/low-light/blur/occlusion 是 physical corruption，默认 `image_valid_mask=true`；frame dropout/burst missing 是 missing/invalid，必须写 `image_valid_mask=false` 或等价 mask。`image_observability_score` 统一表达可用性，不能把遮挡当作整帧缺失，也不能把缺失帧标成 clean corruption。

5. **observability-aware fusion 是窄模型模块，避免污染通用 early-concat fusion。**

   新模块接收已投影到同维度的 `z_img`、`z_gps` 和 reliability metadata，输出 `z_fuse`、weights、gating diagnostics。普通 supervised fusion、CLS-token transformer fusion 和 JEPA query-pool baseline 不被语义替换；Scenario D manifest 显式选择该模块或对应派生配置。

6. **JEPA temporal fallback 只作为 downstream extension。**

   下游 image encoder 可以配置 `temporal_context_encoder(image_history[t-4:t-1])`，在 image degraded/missing 或 `C3/C4 + D3/D4/D6/D7` 条件下预测当前 `z_img[t]`。该路径不读取未来帧，不移动 target，不要求重新定义 Stage 1 JEPA。

7. **输出采用 ignored output root 下的 `results/` 和 `plots/` 子目录。**

   默认输出根建议为 `outputs/analysis/scenario_d_image_observability/<run_id>/`。在该根内生成用户指定文件名：`results/scenario_d_image_observability.csv`、`results/heatmap_cx_dy.npy`、`plots/robustness_surface.png`、`plots/phase_transition_curve.png`、`plots/modality_dominance.png`。这样文件名符合研究需求，同时不污染仓库根目录。

## Risks / Trade-offs

- 图像退化若直接作用于已 normalized tensor，视觉效果可能不等同真实天气/光照。→ 在 metadata 中记录 `input_space`，优先保持 deterministic、shape-preserving 和 metric comparability；后续可替换为更物理的 pre-normalization transform。
- Observability-aware fusion 可能让 JEPA 和 CNN 对照不再只差 encoder。→ Benchmark 必须保留 naive/standard fusion 对照，并在模型组 metadata 中记录 fusion type，避免把架构差异误写成纯 representation 差异。
- `D7` 同时包含 GPS 和 image 条件，容易与 Cx-Dy matrix 重复表达。→ `D7` 定义为 image-side worst-case preset，同时 benchmark 对 `C3/C4 + D7` 记录重点 worst-case；manifest 中必须清楚标注 C 和 D 条件来源。
- Temporal fallback 若实现不慎会引入未来泄漏。→ 单元测试用 toy sequence 断言 fallback source index 仅来自 `t-4:t-1`，并把 source range 写入 metadata。
- 完整 5x8x5 模型矩阵计算成本较高。→ 提供 smoke manifest、evaluation-only manifest 和正式 manifest 分层；测试使用 synthetic/mock batch，不读取真实 `dataset/`。

## Migration Plan

1. 先实现 schema/operator/preset 和 synthetic tests，确认 target/sample metadata 不变。
2. 扩展 benchmark manifest 解析和 aggregation，在 mock/evaluation-only 路径生成 Cx-Dy 表、heatmap NPY 和图表占位/真实图。
3. 实现 observability-aware fusion 与 JEPA downstream temporal fallback，并添加 focused forward/gating tests。
4. 增加配置 preset、CLI help/manifest smoke 和主线文档索引。
5. 运行 `openspec validate add-scenario-d-image-observability-benchmark --strict`、focused pytest 和 CLI help smoke；真实 benchmark 只在用户显式提供数据与 checkpoint 后运行。

Rollback 策略：新增配置和 operator 可通过禁用 Scenario D manifest 回退；现有 clean training/evaluation 默认不启用新 profile，因此旧 workflow 不受影响。

## Open Questions

- 正式 `D1_weather` 的 severity 默认值采用单点 `0.5`，还是 sweep `0.3/0.5/0.7`？
- `image_observability_score` 的默认公式是否先使用可解释线性扣分，还是引入可学习 reliability head？
- Image-JEPA only 在缺失 GPS 的 Cx-Dy 矩阵中是否仍记录 GPS condition metadata，用于横向对齐但不输入模型？
- Modality dominance ratio 优先来自 attention weights、fusion weights，还是在 attention 不可用时使用 ablation delta 作为 fallback？
