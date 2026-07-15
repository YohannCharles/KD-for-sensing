# 《进度汇报》PPT 修改建议与 T2 后续实验方案

> 审阅对象：仓库根目录 `进度汇报.pdf`，PowerPoint 导出，17 页，生成时间 2026-07-12 21:05（UTC+8）。  
> 审阅口径：以当前源码、OpenSpec 和本地可复现实验产物为准；未审查的本地结果只作为 experimental evidence，不升级正式 claim。  
> 红字定位：PDF 中共有两种红色字体 `#FF0000`、`#C00000`，出现在第 4、5、7、9、14、15 页。

## 一、结论先行

PPT 当前最大问题不是措辞，而是主线已经发生变化：第 7-15 页仍把旧 C2 的 prototype alignment 与 supervised router 作为主要创新，而本轮时序缺失实验主线已选 T2。T2 相对 S1 的唯一新增机制是训练期的 confidence-gated temporal superset KL；推理结构、参数量和 FLOPs 与 S1 相同，但训练期需要额外一次在线 teacher forward。

建议把主叙事改为：

1. 真实部署不仅会整模态掉线，还会在历史窗口内逐帧丢失、延迟或不可用；
2. S1 先对每个模态做 masked temporal mean，再做 supervised modality routing；
3. T2 用同一样本的较完整历史作为 stop-gradient teacher：teacher 预测正确时激活约束，熵越低权重越大；
4. seed1 固定 mask screening 支持 T2 作为后续验证主线，但尚不足以形成正式统计结论；
5. prototype 与 circular geometry 不能继续写成已证实的核心贡献，必须由 linear-Gaussian 和 classifier/prototype-loss-off 对照重新验证。

## 二、红字逐页修改

### 第 4 页：明确研究立足点

红字原文：`修改下，明确立足点`

建议直接替换右侧问题框：

> **研究立足点：部署侧连续感知流并不总是完整。** 相机、雷达、LiDAR 和 GPS 既可能整路不可用，也可能因遮挡、延迟或丢包仅缺失历史窗口中的部分帧。本文统一建模“模态 × 时间”可用性，并研究在固定 5 帧历史窗口内，如何利用剩余观测稳定预测下一时刻波束。

页面同步修改：

- 标题由“为什么缺失模态重要”改为“为什么要研究模态与时序缺失”；
- 删除或解释 `mmWave sensing power`。第 12 页的模型输入只有 Image/LiDAR/Radar/GPS，若 mmWave power 只是生成 beam label 的监督来源，应写“64-beam sweep 仅用于生成目标标签，不作为模型输入”；
- 不要写“full modality 不能优化”这类绝对句，改成“只优化 full modality 无法覆盖部署侧不完整感知流”。

### 第 5 页：相关工作与研究缺口

红字原文：

- `缺少 beam 邻近结构。3-5篇(列出文献)，讲讲相比于通用模态缺失，讲讲考虑到beam信息`
- `不直接利用 beam 原型结构`
- `概括下现有模态缺失下的波束预测方法是怎么做的，A(模块)+B(模块)。`

红字属于编辑指令，必须全部从正式 PPT 删除。建议把页面改为四列对比表：`方法 / 缺失处理模块 / 融合或对齐模块 / 与本文的差异`。

可列文献：

| 文献 | 方法概括 | 与当前 T2 的差异 |
| --- | --- | --- |
| Yao et al., [*Robust Multimodal Beam Prediction With Missing Modality*](https://doi.org/10.1109/LWC.2025.3591611), IEEE WCL 2025 | available-modality masking + imputation；channel-attention feature fusion | 主要处理整模态组合，未以同一样本的较完整历史选择性监督缺帧视图 |
| Wen et al., [*AMBER: An Adaptive Multimodal Mask Transformer for Beam Prediction with Missing Modalities*](https://arxiv.org/abs/2512.11331v2), arXiv v2 | missing-aware mask/token + modality/fusion Transformer；CMA + temporal positional embedding | 架构更重，关注 missing-modality token 传播和表征对齐，不是零推理开销的同模型 superset consistency |
| Deng et al., [*Toward Reliable Multimodal Beam Prediction ... via Probabilistic Embedding and Uncertainty-Aware ...*](https://doi.org/10.1109/JIOT.2025.3641184), IEEE IoTJ 2026 | probabilistic embedding + supervised contrastive learning；composite fusion 用 entropy/divergence/confidence 做 sample-specific weighting | 依据分支自估计不确定性加权；T2 的置信度用于决定 teacher 是否值得蒸馏，不直接替代 router |
| Zhu et al., [*Advancing Multi-modal Beam Prediction with Cross-modal Feature Enhancement and Dynamic Fusion Mechanism*](https://doi.org/10.1109/TCOMM.2025.3548021), IEEE TCOM 2025 | MLDA + cross-modal feature enhancement + uncertainty-aware dynamic fusion | 重点是跨模态增强与动态融合；当前 T2 研究的是同模态历史的 partial/superset 一致性 |

建议正文：

> 现有缺失模态波束预测大致采用两类路线：一类通过 mask、补全或完整到缺失的特征对齐缓解表征漂移；另一类依据熵、置信度、分支分歧或 attention 动态融合可用模态。它们主要按整模态失效建模，对同一模态在短历史窗口内仅部分帧缺失的研究较少，也通常未检验“更完整输入是否真的更可靠”。本文因此引入 T2：较完整历史 teacher 预测正确时，对缺帧 student 施加 soft-logit 一致性，并让低熵 teacher 获得更高权重。

注意：不要笼统写“现有方法不利用 beam 信息”。这些方法本身都在做 beam prediction；更准确的缺口是“未验证 dataset-specific beam 邻接几何是否进入缺帧一致性约束”。

### 第 7 页：模型概览与创新点

红字原文：`创新点1：`、`创新点2：`、`融合特征空间`、`每个单模态分支`

当前页面信息过载，并且仍在讲旧 C2。建议拆成两页。

第一页标题：`S1：模态内时序聚合 + 模态间可靠性路由`

- 输入：`[B, 5, 4]` 模态-时间可用 mask；
- 每个模态只聚合其有效历史帧；
- 聚合后的四个分支进入现有 supervised router；
- prototype/router 只标为 S1 的基础组件，不再写成 T2 新增创新。

第二页标题：`T2：置信门控的较完整历史一致性训练`

建议替换两处“创新点”：

> **核心设计 1：双视图时序缺失训练。** Student 接收采样后的缺帧历史，stop-gradient teacher 接收同一样本、同一目标的较完整历史；两个视图不使用未来帧。

> **核心设计 2：选择性 temporal superset KL。** 只有 teacher 预测正确时才激活约束，并以 `1 - normalized entropy` 作为样本权重；当前 `temperature=2`、`KL weight=0.2`。推理阶段删除 teacher 分支，因此 T2 相对 S1 没有新增参数或 FLOPs；代价发生在训练期，已测吞吐由 S1 的 48.16 降到 T2 的 37.10 samples/s（约慢 23%），seed1 墙钟由约 75.2 增至 93.9 分钟（约增加 25%）。

图中红色标签建议改为：

- `较完整历史视图（teacher，stop-gradient）`
- `缺帧历史视图（student）`

### 第 9 页：beam 物理意义与 linear/circular

红字原文：

- `先用自己的话说清楚，不要上来就发明一个新的词`
- `强调beam物理意义，应该是直线，而不是一个圈，请你确认下当前 Beam Prototype Alignment Loss 是约束成1个圈还是一条线，哪个更符合 beam物理意义`

建议先用普通表述：

> Beam 编号不是完全无序的类别。对于按指向角排序的码本，相邻编号通常对应相近的空间方向，因此训练目标可以给真实 beam 附近的码字更高概率；但首尾是否相邻必须由具体码本定义，不能从“64 类”直接假设成圆环。

当前实现的准确答案：

- prototype bank 是 64 个自由学习、归一化后用于相似度分类的向量，没有被硬约束排成圆或直线；
- 当前 S1/T2 的 Gaussian soft target 使用 `beam_label_circular=true`，即距离为 `min(|i-j|, 64-|i-j|)`，因此 0 与 63 被当成相邻；
- 当前 Within@3、ADBA、MAE 和 T1 risk 也使用 circular distance；
- 现有 `c2_no_circular_soft_targets` 同时切成 one-hot，不能回答 linear-Gaussian 与 circular-Gaussian 哪个更好。

DeepSense Scene31-34 本地审计：

| 证据 | 结果 | 解释 |
| --- | ---: | --- |
| 有限 64-beam power sweep | 18,479 | 188 个含非有限值的 sweep 被排除 |
| 0-based label 与 power argmax 一致率 | 100% | 标签顺序可由真实 sweep 检查 |
| 普通相邻 beam 跨样本功率相关均值 | 0.955 | 相邻列高度相关 |
| beam 0 与 beam 63 相关 | 0.637 | 明显低于普通相邻列 |
| 唯一 beam0 最优样本中 beam1 / beam63 的归一化功率 | 0.995 / 0.439 | 端点响应不表现为相邻 |
| 18,181 个同序列、原始连续且两帧均有限的相邻对中的 0↔63 跳转 | 0 | 严格连续帧口径下未观察到端点跳转 |

审计方法：输入为 `scenario31-34.csv` 的 `unit1_pwr_60ghz` 所引用的 64 维 `mmWave_power_*.txt`，以 `(scene, resolved power path)` 去重；18,667 条路径均唯一且无缺文件。剔除 188 条含非有限值的 sweep 后，相邻相关取 18,479 个原始 power 各列跨样本 Pearson 相关矩阵的一阶对角均值；CSV `unit1_beam - 1` 与 power argmax 只在 finite 子集比较。按 `(scene, seq_index)` 的原始 `index` 严格相邻且两帧均 finite 时有 18,181 对；若先丢坏帧再连接保留观测则有 18,363 对，两种口径的 `|delta|=63` 都为 0。完整审计 provenance 写入 ignored `outputs/t2_beam_geometry_head_v1/beam_geometry_audit.json`。

证据边界：finite 数据的 0-based label 范围为 0-62，beam0 仅 1 次最优，beam63 从未最优。上述结果支持“linear 是当前数据上更保守的实验假设”，但端点覆盖太稀疏，不能写成已经证明物理码本必然不环绕。

因此，本 PPT 应把 DeepSense31-34 画成水平有序 beam/扇区，并写“linear codebook assumption is the more conservative hypothesis for the local sweep audit”。圆环只保留为历史实现对照，不能再表述为已确认的物理事实。MMW 的码本需按其 metadata 单独判断。

### 第 14 页：用实际观察替换占位符

红字原文：

- `观察结论：点1 image模态最强点2… 点3… …`
- `此处可补充个可视化（t-SNE/PCA?）`

建议主结果页改成 T2 vs S1 的 temporal missing rate 曲线或表：

| 时序缺失率 | S1 Top1 | T2 Top1 | T2-S1 |
| ---: | ---: | ---: | ---: |
| 0% | 0.5283 | 0.5375 | +0.924 pp |
| 20% | 0.5108 | 0.5165 | +0.563 pp |
| 40% | 0.4997 | 0.5030 | +0.332 pp |
| 60% | 0.4844 | 0.4870 | +0.260 pp |
| 80% | 0.4483 | 0.4550 | +0.664 pp |

建议结论句：

> 在 seed1、每档 4 个固定 cache entry 的筛选中，T2 在五个时序缺失率上的 Top1 均高于 S1，16/20 个 cache entry 更优；去重后为 12/16 个 distinct mask 更优、1 个持平、3 个变差，五档均值提升 0.549 个百分点。该结果支持 T2 作为多随机种子验证主线，但重复 entry 不是独立证据，当前结果还不能形成正式统计结论。

必须同页保留限制：

- T2 五档 Top3 均值比 S1 低约 0.191 pp；
- T2 MAE 由 S1 的约 1.225 变为约 1.263，当前 circular 口径下更差；
- 整模态缺失 avg Top1 基本不变，说明收益主要针对时序缺帧，不是所有缺失条件全面改善；
- 训练末轮聚合诊断中的 gate active ratio 为 42.44%，平均 gate weight 为 0.0467，说明 T2 只约束一部分 teacher 可信样本；它不是 best checkpoint 上重新评估得到的 gate，后续 entropy-gate-error 图需另跑配对离线诊断。

不建议把普通 t-SNE 作为 T2 的核心证据。T2 直接约束 logits，更合适的图是：

1. 按 `mask_index + mask_type + mask_digest` 对齐，并按 digest 去重后的 paired `Delta Top1`；
2. `teacher entropy - gate weight - student error` 散点；
3. 若必须展示表示空间，用 PCA 画同一样本 teacher/student 的配对位移，并放附录。

若保留旧 router 权重热图，需补充三点：image 可用时通常占主导；image 缺失时权重转移到其它可用分支；single-modality 行权重为 1 是 masked softmax 的必然约束，不是模型学到可靠性的证据。

### 第 15 页：删除“未来补充一个图”

红字原文：`未来补充一个图`

正式汇报不能保留占位说明。两种处理任选其一：

- 若本轮没有真实连续轨迹结果，直接删除此页；
- 若后续完成 MMW 轨迹评估，使用固定四种模态颜色，画三层时间轴：上层为逐帧可用性，中层只画推理时存在的 router 模态权重，下层为真实/predicted beam；标出 3 个代表时刻的具体数值。

不要使用“RGBA 炫彩融合”表达权重，混色不利于比较。固定四色堆叠面积图或四条折线更清晰。图标题应为“沿轨迹的模态路由权重”，避免误解成 T2 学习了逐帧 attention。

T2 的 confidence gate 是训练期离线诊断量，依赖 ground-truth correctness、superset teacher entropy 和较完整历史，部署推理时不存在。若另页展示，必须标注“离线 teacher-student 诊断（需标签与完整历史）”，不能与轨迹 router 权重画成同一种在线信号。

## 三、非红字但必须同步修改

- 第 1 页：标题改为“面向 6G 多模态感知辅助波束预测的模态与时序缺失鲁棒性研究”，更新汇报日期。
- 第 3 页：与第 2 页重复，可删除。
- 第 8 页：编号应为 2.2；补一句“S1 与 T2 的编码器和推理图完全一致”。
- 第 9 页：右栏文字越界且过密，拆页；方法名放在普通语言解释之后。
- 第 10 页：标为“S1 基线中的模态路由”或移到附录，不能写成 T2 新增模块。
- 第 12 页：补充 split、seed、history/prediction window、checkpoint 选择、训练缺失率、固定 mask 数和主指标。
- 第 13 页：旧图是整模态 Full/Drop1/2/3，标为“阶段 1：C2 整模态缺失结果”；不能支撑 T2 时序缺失结论。“缺失越严重时越稳”改为“相对对照的优势在部分严重缺失条件下扩大”，因为绝对性能仍然下降。
- 第 14 页：“prototype 削弱 GPS shortcut”目前只是机制假设。若要验证，应做 GPS shuffle 或跨场景检验，t-SNE/PCA 不能证明因果。
- 第 16 页：编号顺序错误且第 5 项 `c` 未完成；历史窗 5/预测窗 1、时序缺失预处理和 seed1 消融已经完成，不应继续列作未来工作。

## 四、当前状态

### 已完成

- history window=5、prediction window=1 已固定；
- 模态-时间粒度缺失、零填充、mask metadata、固定 eval cache 已实现并通过测试；
- S1、T1、T2、A1、A2、A3、T1+T2、J1 seed1 已在 GPU0-7 完成训练与固定 mask 评估；
- T2 是五档 temporal missing mean Top1 最优的 seed1 候选；
- 选择 T2 是按五档 mean Top1，而不是所有指标全面占优：T1 的 Drop80 Top1 为 `0.457419`，高于 T2 的 `0.454965`；T1 whole-modality avg Top1 为 `0.419870`，也高于 T2 约 `0.405930`，且 T1 的 MAE/ADBA 更好。T1 因而保留为严重缺帧和整模态缺失支线，不与当前 T2 主线合并训练；
- A2 失败；J1 主指标和 Drop80 为负，不运行 J2；
- MMW sunny/rainy/foggy 的 5 个 Town03 场景 H5/P1 本地产物已生成，但 rainy/foggy 的权威 `data_availability.json` 仍登记为 pending；availability writer 尚未识别 `_h5p1` metadata/sanity 文件，且当前没有 T2 多天气主配置或训练结果，不能写成工作流已 ready；
- final C2 prototype/head package 消融已有三 seed 本地结果；classifier/prototype-loss-off package 相对 prototype 主配置的 avg-missing Top1 约 `+1.376 pp`。该 package 同时切换 head、关闭两类 prototype loss 并关闭 prototype-margin router feature，因此只能说明当前 prototype/head package 未获支持，不能归因为 prototype 单一因素；prototype 贡献仍需隔离验证。

### 仍缺少

- 当前 T2 相对 S1 的 seeds2/3 稳定性；
- pure linear-Gaussian 与 circular-Gaussian 的公平对照；
- T2 在 classifier/prototype-loss-off head 下是否仍有效；
- 三 seed按 mask digest 去重的 paired-mask 证据；逐样本 paired evidence 属于后续 claim-grade 工作，不在本轮范围内；
- T2 gate 的定量可视化和真实连续轨迹案例。

## 五、立即执行的详细方案

### 第一轮：8 卡并行

| GPU | 作业 | 目的 |
| ---: | --- | --- |
| 0 | S1 circular-prototype seed2 | current baseline 多 seed |
| 1 | S1 circular-prototype seed3 | current baseline 多 seed |
| 2 | T2 circular-prototype seed2 | 当前 T2 多 seed |
| 3 | T2 circular-prototype seed3 | 当前 T2 多 seed |
| 4 | S1-LG seed1 | linear-Gaussian 匹配基线 |
| 5 | T2-LG seed1 | 检验 linear 几何下 T2 |
| 6 | S1-CLS seed1 | classifier 匹配基线 |
| 7 | T2-CLS seed1 | 检验关闭主动 prototype 依赖后 T2 |

资源固定为每卡一进程、batch64、4 workers、prefetch2、persistent workers、PyTorch intra-op12/inter-op1。训练和评估分别使用独立 ignored roots，避免覆盖原 seed1 manifest。

`CLS` 表示行为上的 classifier/prototype-loss-off：主动 prototype alignment、modality prototype loss 和 prototype-margin router feature 均关闭；共享模型仍为 checkpoint 兼容而实例化一个不参与 logits/loss/router 的 unused `prototype_bank`，因此不能声称参数结构或 checkpoint 已彻底移除 prototype。

权威第一轮 roots 为 `current_multiseed/` 与 `candidate_screen_clean/`。最初的 `candidate_screen/` 因 router oracle 仍残留 circular distance 已标记 `invalidated/killed`，禁止 resume、eval、summary 或引用其中任何产物。

### 第一轮候选门禁

LG 或 CLS 候选只有同时满足以下条件才补 seeds2/3：

1. T2 variant 相对匹配 S1 variant 的五档 mean Top1 为正；
2. Drop80 delta 为正；
3. Drop0 delta 不低于 `-0.005`；
4. T2 variant 五档 mean Top1 相对 current T2 不低于 `-0.005`。

若 linear 与 circular Top1 差距不超过 0.2 pp，优先采用物理口径更可信的 linear。候选失败时记录具体 skipped reason，不继续消耗多 seed 算力。

### 第二轮：只补晋级候选

若 LG 与 CLS 都通过，则 GPU0-7 分配为：

| GPU | 作业 |
| ---: | --- |
| 0 | S1-LG seed2 |
| 1 | S1-LG seed3 |
| 2 | T2-LG seed2 |
| 3 | T2-LG seed3 |
| 4 | S1-CLS seed2 |
| 5 | S1-CLS seed3 |
| 6 | T2-CLS seed2 |
| 7 | T2-CLS seed3 |

若只通过一个候选，只运行该候选及匹配 S1 的四个作业，不用无关实验填满 GPU。

### 固定评估与最终判定

- split：Scene31-34 validation，`stratified_80_10_10`，split seed42；
- checkpoint：每个 run 的 best Top1；
- temporal missing rates：0/20/40/60/80%；
- fixed cache：所有方法、seed 共享相同 cache；paired 汇总记录 `mask_index`、`mask_type`、`mask_digest` 并按 digest 去重，重复 entry 不作为独立证据；
- 主门禁统计：继续使用冻结协议中每档 4 个 cache entry 的 matrix 均值；digest 去重后的 paired-mask delta 单独报告，只作为辅助稳定性证据，不替代主门禁；
- 主选择：五档 mean Top1、Drop0-60 mean Top1、Drop80 Top1；
- 辅助指标：Top3、Within@3、ADBA、MAE、gate active ratio/mean；
- circular 与 linear 的 ADBA/Within@3/MAE 分开汇总，跨几何只直接比较 Top1/Top3。

### 复现与产物路径

- 历史 seed1 训练：`outputs/s1_lightweight_temporal_robustness_v1/{S1,T2}/seed1/`；
- current seeds2/3 训练：`outputs/t2_beam_geometry_head_v1/current_multiseed/`；
- 第一轮 LG/CLS seed1：`outputs/t2_beam_geometry_head_v1/candidate_screen_clean/`；旧 `candidate_screen/` 已作废，禁止使用；
- config-mode 评估：`outputs/t2_beam_geometry_head_v1/eval_config_mode/`；current checkpoint 强制 linear 复算：`outputs/t2_beam_geometry_head_v1/eval_current_linear/`；
- 首轮 strict 汇总：`outputs/t2_beam_geometry_head_v1/summary_first_round/`；current-linear 汇总：`outputs/t2_beam_geometry_head_v1/summary_current_linear/`；
- 条件式晋级训练：`outputs/t2_beam_geometry_head_v1/candidate_advance/`，只在 seed1 门禁通过后生成；
- checkpoint 统一选择每个 run 的 `checkpoints/best_top1.pth`，不能在训练未结束时把中途 best 当作冻结结果；
- 固定 cache：`outputs/temporal_eval_masks_s1_lightweight_rate_v1/`，`seed=20260708`、`history=5`、每档 4 entries；五档 checksum 依次为 `a0c591043731875a`、`699d14abccfacc95`、`be537052d6b90eb4`、`6e0d2e9aa96b44e3`、`b55a3b6fe7db10e9`；
- metric mode：current config-mode 为 circular，LG/CLS config-mode 为 linear；距离敏感指标禁止跨 mode 聚合。

T2 最终保留为实验主线的最低条件：

1. 三 seed 的五档 mean Top1、Drop0-60 mean、Drop80 平均均高于匹配 S1；
2. 至少 2/3 seed 的五档 mean delta 为正；
3. 三 seed 平均 Drop0 delta 不低于 `-0.005`；
4. 结果继续标为 local experimental；本轮完成 paired entry 与 digest 去重的 paired-mask review，正式 claim 仍需后续逐样本 paired evidence review。

## 六、建议替换第 16 页“下一步工作”

1. **完成 T2 多随机种子验证**：S1/T2 seeds1-3，同 split、同 mask、同 checkpoint 规则。
2. **核验 decision-space 假设**：公平比较 circular-Gaussian、linear-Gaussian 与 classifier/prototype-loss-off。
3. **补齐配对统计**：输出逐 seed mean/std、同 mask delta 和高缺失率/Drop0 guardrail。
4. **生成 T2 专属可视化**：rate 曲线、paired-mask delta、teacher entropy-gate-student error；真实轨迹图只用已验证 MMW checkpoint。
5. **冻结 DeepSense 主线后再进入 MMW weather**：分别建 sunny/rainy/foggy 训练/测试协议，不与本轮 T2 几何/head 结论混写。

## 七、执行记录

| 阶段 | 状态 | 产物/结论 |
| --- | --- | --- |
| PDF 红字与页面审阅 | 已完成 | 17 页、6 个含红字页面 |
| T2/S1 seed1 分析 | 已完成 | T2 五档 Top1 均为正增益，仍有 Top3/MAE caveat |
| DeepSense beam geometry 审计 | 已完成 | 本地数据更支持 linear，不支持首尾相邻的物理解读 |
| OpenSpec 与最小实验入口 | 已完成 | `validate-t2-beam-geometry-and-head` 已通过 strict validation 与 focused tests |
| 第一轮 GPU0-7 | 已停止并收口 | LG/CLS 四候选 seed1 已完成；current S1/T2 seeds2/3 后续按用户的 MMW 资源切换决策在 epoch 5/6 停止 |
| 第二轮晋级实验 | 已取消 | 第一轮缺少完整 current 对照和五档固定 temporal matrix，门禁为 `unavailable/cancelled`，不补候选 seeds2/3 |
| 三 seed 最终汇总 | 不可用 | 不完整 checkpoint 不参与汇总；本 change 不形成 T2、geometry 或 head 的三 seed claim |

收口边界：`candidate_screen_clean/` 的四个 seed1 run 可作为 local experimental 实现/训练证据；`current_multiseed/` 中止 run 的 stale `running` status、5/6 epoch metrics 和 checkpoint 只能用于审计停止事实，禁止解释为完成训练。后续资源已转向 MMW all-weather 主线，因此不恢复旧 DeepSense 训练或评估编排器。
