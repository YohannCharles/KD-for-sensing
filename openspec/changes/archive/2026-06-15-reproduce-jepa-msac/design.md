## Context

本 change 复现 arXiv:2603.29796 v1《JEPA-MSAC: A Joint-Embedding Predictive Architecture for Multimodal Sensing-Assisted Communications》。论文使用 DeepSense 6G Scenario 32，基于 Image、Radar、LiDAR、GPS 和 RF beam-level RSRP 历史序列做 temporal block-masked JEPA 预训练，再冻结 backbone 训练 localization、beam prediction 和 RSSI prediction heads。

仓库当前已有 Image+GPS GPS-conditioned JEPA、JEPA downstream pooler、DeepSense6G 多模态 dataset、模块化模型 registry、paper/workflow baseline 边界、结果账本和本地产物边界。现有 JEPA 实现只覆盖 Image+GPS patch latent prediction，不覆盖多模态 token space、RF 历史 token、future latent slot inference 或三任务 heads，因此本复现需要作为新的 paper/workflow family 落地，而不是把所有行为塞进现有 `gps_conditioned_jepa`。

## Goals / Non-Goals

**Goals:**

- 提供 JEPA-MSAC local reproduction workflow：Stage 1 自监督预训练、Stage 2 冻结 backbone 多任务 head 训练、report/ablation 汇总。
- 对齐论文默认协议：Scenario 32、13 帧滑窗、`T_hist=8`、`T_pred=5`、64 beams、70/30 随机 split、token counts `9/16/16/1/1`、mask ratio `0.5`、Stage 1 `100` epochs、Stage 2 `30` epochs，并提供 smoke 配置。
- 复用当前 `src/kd_sensing` 包结构、registry、engine extension、dataset transform、objective metadata、runtime metadata、文档账本和 ignored output 目录。
- 支持无真实数据的单元/合成 smoke 测试，同时允许用户在本地 DeepSense6G Scenario 32 上运行长实验。
- 明确 claim status：未实际完成长训练前只能标记为 unverified/local-ready，不能把 smoke 或 mock 结果写成论文复现结论。

**Non-Goals:**

- 不提交 DeepSense6G 原始数据、生成 cache、checkpoint、TensorBoard event、训练日志、图表或 metrics CSV。
- 不恢复 legacy KD、HiST/Hist、standalone Top8 selector、GPS residual、camera residual、Raymobtime s008 或 Multimodal-NF 入口。
- 不承诺逐像素复刻论文 Figure 2 图形，也不把论文未公开的随机种子、私有预处理细节或未给出的官方源码假定为已知。
- 不把 RF 声明为新的 canonical modality；RF beam-level RSRP 历史通过现有 mmWave/beam-power target schema 和 workflow 专用字段表达。

## Decisions

### Decision 1: 使用 paper/workflow baseline 路径

JEPA-MSAC 包含两阶段训练、EMA target encoder、future latent slot 推理、多任务 heads、专用指标和 Table/ablation 报告，属于 `model-architecture-extension-contract` 中的 workflow/paper reproduction。实现放在 `src/kd_sensing/baselines/jepa_msac/` 和包内 CLI，例如 `kd_sensing.cli.run_jepa_msac`，通用组件放在 `models/`、`engine/`、`losses/`、`evaluation/` 的窄模块中。

替代方案是注册一个普通 `modular_sequence` 配置。这个方案无法自然表达 Stage 1/Stage 2、EMA target encoder、future mask slots 和冻结 head training，也会把 workflow orchestration 泄漏到通用 trainer。最终选择 workflow 路径，但模型内部仍复用 registry 和 `ModelOutput` 适配，避免复制训练循环。

### Decision 2: 新增 JEPA-MSAC 模型组件，而不是改造 GPS-conditioned JEPA

新增 `jepa_msac` 相关模型模块，包含 modality tokenizers、factorized positional embedding、context/target transformer encoders、mask token predictor 和 task heads。现有 `gps_conditioned_jepa` 继续服务 Image+GPS 预训练与 downstream reuse；JEPA-MSAC 可复用其中的 EMA、mask/loss、metadata 写出和 checkpoint 抽取模式，但不改变其输入/输出契约。

替代方案是在 `GPSConditionedJEPA` 中扩展多模态分支。这样会让 Image+GPS 既有测试和配置承受额外 shape/field 复杂度，还容易破坏当前 downstream pooler 语义。隔离新模型能把 paper-specific 行为锁在可审计边界内。

### Decision 3: RF 历史作为 workflow 专用 beam-power 输入

论文中的 RF token 是历史 beam-level RSRP vector `x_RF[t]`。仓库已有 `mmwave` modality、beam power/label calibration 和 objective target schema，因此 workflow 数据层新增 `rf_power_history` 或等价字段，来源映射到 DeepSense6G 当前可用的 beam power / mmWave RSRP 数据；runtime metadata 记录 `paper_modality: RF` 与仓库字段的映射。

替代方案是把 `rf` 加入中心化模态契约。当前 `modality-contracts` 固定 canonical modality 顺序，新增 canonical modality 会影响配置解析、batch runtime 和诊断。为避免大范围破坏，先采用 workflow-local alias，并在 spec 中要求不得把 `rf` 暴露为新的通用 modality。

### Decision 4: 数据协议使用 manifest 化本地复现

新增 Scenario 32 JEPA-MSAC split/window manifest builder，记录原始 CSV、scene、sample rate、window length、history/prediction split、随机 seed、70/30 分割、启用模态、target schema 和 checksum/digest。真实数据读取仍走现有 DeepSense6G dataset/transform helper；测试使用 synthetic fixture 或 metadata-only manifest。

替代方案是直接在 runner 中随机切分 dataframe。manifest 化能让 Stage 1、Stage 2、evaluation 和报告使用同一 sample universe，避免长实验后无法解释 split 差异。

### Decision 5: Stage 1 和 Stage 2 共用 runner，但 stage 可独立执行

提供一个包内 CLI，支持 `--stage pretrain`、`--stage heads`、`--stage evaluate`、`--stage report` 和 `--stage all`，并支持 `--dry-run`/smoke。Stage 1 输出 checkpoint 与 pretraining metadata；Stage 2 从 checkpoint 加载 context encoder/predictor，冻结 backbone，只训练 heads；report 只读本地产物并写 summary 到 ignored 输出。

替代方案是多个独立 CLI。单入口能共享 config、manifest、resume 和 report schema，减少入口 allowlist 维护成本；stage 参数仍保留独立调试能力。

### Decision 6: 指标和 claim 分层

实现 representation metrics `RRankMe` 和 `RLDA`，任务指标 ADE/FDE、Top-1/Top-3、L1-RSRP diff、RSSI RMSE/MAE，并输出 horizon-wise 与 aggregate 表。`docs/result_claims_registry.md` 只记录路径、摘要和 claim status；只有 paper-aligned 长训练完成并审计后才能标记为 local strict-validation 或 local reproduction。

替代方案是只输出训练日志中的 loss/accuracy。论文复现需要对齐多个任务与 latent quality，缺少这些表会让结果不可比较。

## Risks / Trade-offs

- [Risk] 论文未公开官方源码或完整随机种子，细节如 EfficientNet 版本、radar/LiDAR exact preprocessing、RSSI target 定义可能存在歧义。→ Mitigation：metadata 记录每个解释决策，结果账本使用 `local substitute`、`unverified` 或 `blocked detail` 状态，不宣称 official reproduction。
- [Risk] 多模态 token 序列和 EfficientNet/Transformer backbone 显存开销较高。→ Mitigation：提供 smoke/lowmem 配置、可配置 latent dim/token counts/batch size、epoch_subsampling 和 no-real-data unit tests。
- [Risk] DeepSense6G Scenario 32 本地数据字段可能缺少论文所需 RF beam-power vector 或 radar/LiDAR 原始格式不一致。→ Mitigation：manifest builder 做只读 audit，缺字段时输出 blocked reason；不生成伪 target。
- [Risk] 新 workflow 可能诱发从通用 trainer 复制训练循环。→ Mitigation：tasks 要求复用 training extension、checkpointing、runtime metadata、evaluation/report helper；workflow 只负责 stage orchestration。
- [Risk] 文档账本和配置 allowlist 漂移。→ Mitigation：同步 `docs/project_surface_inventory.md`、架构边界测试和 CLI help smoke。

## Migration Plan

1. 先落 spec、tests 和 synthetic fixtures，确保 registry、mask、loss、dataset manifest、CLI help 与 report schema 可验证。
2. 实现模型组件和 Stage 1 smoke 训练，确认 EMA、masked SmoothL1、checkpoint/resume metadata。
3. 实现 frozen inference 与 Stage 2 heads，确认 backbone 参数冻结、heads 可训练、三任务输出和 metrics。
4. 增加 paper-aligned 配置和文档账本条目，默认状态为 `unverified/local-ready`，等待用户本地长训练产物。
5. 若需要 rollback，删除新增 `jepa_msac` 模块、CLI script entry、configs、tests 和 docs 行即可；现有 JEPA/Image+GPS 与 BGAM 入口不受影响。

## Open Questions

- 论文未给出的随机 seed 是否由本仓库选择固定默认值，并在结果中标记为 local seed？
- RSSI loss 的 target 应使用 scalar RSSI、beam-wise RSRP vector 均值，还是完整 beam-power profile 的 SmoothL1？实现时需要在配置和 metadata 中显式记录选择。
- 默认 vision tokenizer 是否严格使用 torchvision EfficientNet，还是提供轻量 CNN/VisualPatchTokenEncoder smoke fallback？
- 若本地 Scenario 32 radar/LiDAR cache 已经是预处理特征，是否允许跳过 raw preprocessing 并在 report 中标记为 cache-based local protocol？
