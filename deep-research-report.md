# 知识解耦与轻量化跨场景自适应波束预测方案深度调研

## 研究定位与核心判断

把这个方向定位成“**半年内可完成、适合冲击 IEEE Globecom / ICC、重点在清晰创新点而不是堆大模型**”是可行的，但前提是**不要把论文写成普通的 domain adaptation/fine-tuning 故事**。从 2020–2026 年的代表性工作看，现有文献已经在四条线上铺得比较满：其一是单模态可行性验证，例如 GPS、视觉、LiDAR、Radar 辅助 beam prediction/beam tracking；其二是多模态融合架构升级，例如 multimodal transformer、动态融合、VLM/LLM、量子 transformer；其三是 unseen environment/few-shot transfer，例如 TUNE、MAML、cross-environment transfer learning、DANN-style fine-tuning；其四是缺失模态鲁棒性，例如 missing-modality masking 与 AMBER。真正还没有被系统性打透的，是**在多模态 beam prediction 中，到底什么知识应该跨场景共享，什么知识应该留给新地点的小型 scene-private adapter 去适配**。这正是你这个方向最适合切入的“空白区”。citeturn44view0turn5view3turn5view4turn5view5turn6view0turn6view3turn25view0

就数据与任务而言，DeepSense6G 是一个**超过 100 万数据点、40+ 场景、同步多模态感知与通信**的大规模真实数据集；2022 Beam Prediction Challenge 使用的是 **scenarios 31–34**，并强调**对 unseen scenario 的 generalization**，挑战输入是**相机/LiDAR/Radar 各 5 帧历史样本 + GPS 2 帧历史样本**，目标是预测 **64-beam codebook** 下的最优波束，官方采用 **DBA score** 作为核心评估指标。公开可见的场景信息还显示：**31 与 32 为 daytime street-level V2I 场景，33 与 34 为 nighttime street-level V2I 场景**。这意味着你天然拥有“**新地点迁移** + **昼夜风格转移** + **多模态互补**”三种论文故事线，但审稿最容易认可的仍然是**what to transfer 而不是 how to fine-tune**。citeturn14view0turn17search3turn5view1turn33view0turn17search1turn19search0turn20search3turn38search5

最直接的结论是：**你的最佳投稿叙事不是“我做了一个更强的 multimodal transformer”，也不是“我做了一个 target fine-tuning trick”**。更稳的 Globecom/ICC 叙事应该是：**absolute beam index 对跨场景迁移并不是好的知识单元；更可迁移的是 coarse beam semantics、角度连续性、多模态几何一致性与运动到 coarse group 的映射；而道路几何、基站安装角度、相机 FoV、外观风格、局部反射体分布等应由小型 scene-private adapter 承担。** 这一故事能够同时避开 TUNE/MAML/DANN/Cross-Environment TL 的直接重合，也能避开 AMBER 那类“鲁棒融合/缺失模态”主线。citeturn5view4turn5view5turn6view0turn25view0turn6view3turn33view1

## SOTA查新与去重矩阵

下表把与你方向最相关、且最可能形成“重复嫌疑”的代表性工作压缩成一个去重矩阵。表中“冲突点”不是说不能引用，而是指**如果你的论文标题、摘要和方法卖点跟它落在同一层面，就会被审稿人认为重复**。

|论文|年份|数据集|任务|方法核心|是否使用 DeepSense6G|是否跨场景|是否少标注适应|是否知识解耦|局限性|与我的方向可能冲突点|可避开的创新入口|
|---|---:|---|---|---|---|---|---|---|---|---|---|
|Multi-Modal Beam Prediction Challenge 2022: Towards Generalization citeturn43search0turn44view0|2022|DeepSense6G 31–34|unseen-scenario beam prediction benchmark|官方任务定义 + 位置基线 + DBA 指标|是|是|否|否|更多是 benchmark 与 baseline，不回答“迁移什么”|若你只讲“跨场景 generalization for beam prediction”，会显得只是 challenge follow-up|把 story 升级成“知识分工”而非“泛化更好”|
|Position Aided Beam Prediction in the Real World citeturn36view0|2022/2023|DeepSense6G 1–9|GPS-aided beam prediction|分析 GPS 真实可用性、模型选择与更合理指标|是|部分|否|否|聚焦 GPS 单模态；指出真实 GPS 误差会伤害 beam prediction|若你仍把 GPS→beam 映射当主要 transferable knowledge，会被质疑没有吸收该文结论|将 GPS 视为**coarse prior**，不是可直接迁移的 fine mapping|
|Vision-Position Multi-Modal Beam Prediction Using Real Millimeter Wave Datasets citeturn10search0turn10search10|2022|DeepSense6G 5–6|vision+GPS beam prediction|双模态融合证明视觉+位置优于单模态|是|否|否|否|主要是 feasibility + fusion 增益|若你卖点还是“多模态优于单模态”，没有新意|用它作 lightweight backbone 参考，不把 fusion 当主创新|
|Radar Aided 6G Beam Prediction: Deep Learning Algorithms and Real-World Demonstration citeturn10search2turn10search9|2022|DeepSense6G 实景 V2I|radar-aided beam prediction|雷达信号处理 + DL，降低 beam training overhead|是|否|否|否|单模态，偏 radar feature engineering|若你把 radar encoder 设计写成主创新，会和单模态工作撞车|把 radar 作为 shared/private 分工中的一环，而非主角|
|LiDAR Aided Future Beam Prediction in Real-World Millimeter Wave V2I Communications citeturn42search3turn42search4|2022/2023|DeepSense6G V2I|future beam prediction / tracking|LiDAR 预测当前与未来 beam|是|否|否|否|强调未来 beam tracking，不是新地点适应|若你主打“未来 beam tracking”会偏题|保留 temporal cue，但论文主线仍是跨场景知识迁移|
|Computer Vision Aided Beam Tracking in a Real-World Millimeter Wave Deployment citeturn11search1turn11search11|2022|实网视觉场景|future beam tracking|encoder-decoder 由视觉预测未来 beam|部分|否|否|否|聚焦 tracking 而非 adaptation|若你引入历史 beam/长时预测为主，会被拉到 tracking 文献里|只把短时历史作为辅助，不改变问题主线|
|Multimodal Transformers for Wireless Communications: A Case Study in Beam Prediction citeturn5view3turn5view1|2023|DeepSense6G Challenge|multimodal beam prediction|CNN + layer-wise transformer token fusion；图像+GPS 最优|是|部分|否|否|创新点在 fusion/backbone/augmentation，不是 transfer mechanism|若你方法本质上还是“更好的 multimodal transformer”，会高度冲突|固定轻量 backbone，把创新重心转去 decoupling + adapter|
|TUNE: Transfer Learning in Unseen Environments for V2X mmWave Beam Selection citeturn5view4|2023|FLASH + 公开 real/synthetic 数据|unseen-environment beam selection|选最相似 source model + 动态保留层进行 TL|否|是|是|否|解决“从哪迁移、迁多少层”，但不解释“哪类知识该迁移”|若你方法仍是 layer freezing/fine-tuning 策略，会被认作 TUNE 变体|显式定义 shared coarse semantics 与 private scene geometry|
|Meta-Learning for Image-Guided Millimeter-Wave Beam Selection in Unseen Environments citeturn5view5|2023|FLASH|few-shot unseen beam selection|MAML 学初始化，测试时少量 samples 快速适配|否|是|是|否|重点是初始化与快速更新；单模态图像；非 DeepSense6G|若你卖点是“few-shot unseen adaptation”，容易被归到 MAML story|强调你不是 learning initialization，而是 learning transferable semantics|
|Deep Quantum-Transformer Networks for Multi-Modal Beam Prediction in ISAC Systems citeturn38search10turn40search2|2024|DeepSense6G V2I|multimodal beam prediction / zero-shot testing|量子层 + transformer，追求更强表示与 zero-shot 性能|是|是|否|否|模型较重，创新点在 network family|若你继续堆 exotic backbone，会被问和 QTN 有何本质区别|坚持“普通 backbone + 更合理 transfer unit”|
|Robust Multimodal Beam Prediction With Missing Modality citeturn12view5|2025|未见明确 DeepSense6G|missing-modality beam prediction|随机 masking + missing-modality imputation + channel attention|未明确|否|否|否|主问题是缺失模态鲁棒性，不是跨场景适应|若你把 modality dropout 写成主贡献，会与该线重合|把模态缺失只放进 robustness benchmark，不放进 main claim|
|Multi-Modal Sensing-assisted Beam Prediction using Real-World Dataset citeturn12view0|2025|DeepSense6G/Challenge数据|multimodal beam prediction|ResNet-SE + PIformer，提升精度并降低训练时间|是|未强调|否|否|依然是更强 fusion network|若你 claim “更轻 + 更准的多模态主干”，冲突明显|把轻量化限定在 adapter，不在 backbone SOTA 上较劲|
|Cross-Environment Transfer Learning for Location-Aided Beam Prediction in 5G and Beyond Millimeter-Wave Networks citeturn6view0turn6view1|2025|ray-tracing 两个城市环境|cross-environment transfer learning|reference gNB 模型迁移到 target gNB，5% target fine-tuning|否|是|是|否|位置单模态；核心是 limited-data fine-tuning|若你写“location-aided beam prediction with limited target fine-tuning”，高度冲突|使用多模态 + shared/private + source-free/few-shot adapter|
|A Deep Transfer Learning-Based Low-overhead Beam Prediction in Vehicle Communications citeturn25view0|2025|仿真车联网|DANN-style transfer learning|fine-tuning 时加 domain classifier 抽取 domain-invariant feature|否|是|是|否|典型 adversarial DA；没有 scene-private 建模|若你使用 GRL+domain confusion 作为主招式，会被视作同型工作|只把 DANN 作为 baseline，对比你的“知识分工”更物理合理|
|Advancing Multi-Modal Beam Prediction With Cross-Modal Feature Enhancement and Dynamic Fusion Mechanism citeturn46search2|2025|摘要未明确 DeepSense6G|multimodal beam prediction|MLDA + CMFE + uncertainty-aware dynamic fusion|未明确|未强调|否|否|仍是 fusion/robustness 线|若你主打动态融合/不确定性加权，会被问与该文差别|把 prototype alignment 限定在 coarse semantics，不在 fusion trick|
|When VLM Meets Beam Prediction: A Multimodal Contrastive Learning Framework citeturn12view4|2025|DeepSense6G|multimodal beam prediction|图像-LiDAR 对比预训练 + 位置作为文本 prompt|是|未强调|否|否|创新在 VLM backbone 与跨模态对齐，不是 target adaptation|若你转向 CLIP/VLM prompt，会偏离半年可落地方向|保留 contrastive 为可选 loss，不上大模型|
|AMBER: An Adaptive Multimodal Mask Transformer for Beam Prediction with Missing Modalities citeturn6view2turn6view3|2025/2026|DeepSense6G|missing-modality robust beam prediction|missing-modality-aware mask transformer + CMA + temporal embedding|是|未强调|否|否|主旨是缺失模态与鲁棒融合；模型较大|若你论文强调“开放世界缺失模态鲁棒 beam prediction”，会正面冲突|把 AMBER 仅作为 robustness baseline；主线仍是跨场景知识迁移|

从审稿人视角，**最强重复风险**来自四类工作。第一类是 **TUNE / Cross-Environment TL / DANN-style 组合**：它们已经覆盖了“unseen environment + limited target data + fine-tuning/适配”这条路，但本质上都把 transfer 当作“迁整个表示或者迁部分层”，没有回答**哪些知识该迁移、哪些不该迁移**。如果你的摘要还写成“limited labels in target + adaptation to unseen scenario”，会非常危险。citeturn5view4turn6view0turn25view0

第二类是 **MAML / meta-learning**。它解决的是“快速适配的初始化参数”，不是“具有物理含义的知识分工”。只要你不把核心 claim 写成“few-shot meta-adaptation for unseen beam selection”，并且不把 adaptation 的本体说成“faster convergence from learned initialization”，你就能比较稳地避开它。你的进路应该是：**coarse beam semantics 天生更 transferable，而 fine beam refinement 天生更 site-specific**。citeturn5view5

第三类是 **multimodal fusion 升级线**，包括 Multimodal Transformers、QTN、JCN 2025、CMFE、VLM/VLM-like 方法。它们都在证明“更强的融合网络更准确”，但很少把**新地点 adaptation**做成核心问题。你要明显区别于它们：**backbone 轻量、可复现、可复用；创新点不在 fusion 结构，而在知识解耦和 target-side 小参数更新**。citeturn5view3turn12view0turn38search10turn46search2turn12view4

第四类是 **missing modality robustness**，尤其 AMBER。这条线非常容易把论文带偏，因为一旦你开始强调 modality masking、fusion token、模态缺失训练策略，审稿人会立刻拿 AMBER 对你。最稳妥的做法是：**把缺失模态只放在 robustness benchmark 中，证明你的 decoupling 方法在模态噪声下仍然稳，不把它作为主创新。**citeturn6view3turn12view5

## 科学问题重定义

**Q1：为什么普通的 domain-invariant representation 不适合毫米波 beam prediction？**  
问题定义是：beam prediction 的标签并不是纯视觉分类标签，而是由**传播几何、基站朝向、遮挡结构、局部反射体、设备 FoV 与 codebook 映射**共同决定的结果；因此“全表示域不变”会同时抹掉一部分**必须保留的 scene-specific 判决信息**。物理直觉上，毫米波窄波束强依赖 LOS/NLOS 与环境几何，DeepSense challenge 本身就是为了研究 unseen scenario generalization 而设计；Cross-Environment TL 也明确指出 mmWave propagation具有**highly site-specific** 的特点；而 DANN-style recent work 仍然把目标写成“抽 domain-invariant features”，这对于 beam prediction 这种“粗语义可共享、细映射不可共享”的任务来说过于粗暴。本文的切入点因此不是“让所有特征都 invariant”，而是**只让 coarse beam semantics invariant，让 fine refinement 明确由 private branch 承担**。citeturn44view0turn6view0turn25view0turn24search2

**Q2：为什么 coarse beam semantics 比 absolute beam index 更适合跨场景迁移？**  
绝对 beam index 在不同地点、不同基站安装角、不同 FoV 与不同遮挡物分布下并不稳定；但**相邻 beam 的角度连续性、粗粒度 beam sector/group 的物理语义、以及“运动趋势→coarse beam group”映射**通常更稳定。官方 challenge 之所以采用 DBA 而不仅是 Top-k，就是因为**预测落在相邻 beam 上时，在物理接收功率上并不一定是灾难性错误**；challenge 论文和后续指标分析都强调，Top-k 这种纯分类指标不足以反映 beam 任务的物理语义，DBA 与 power ratio 相关性更高。换言之，**absolute beam class 不是最适合 transfer 的知识单元，coarse beam semantics 才是。** 本文的切入点是把监督从“单层 absolute class CE”改写成“coarse group + fine offset”的层次监督，并只在 coarse group 空间做 prototype alignment。citeturn45view0turn33view1turn38search7

**Q3：为什么新地点适应应该只更新极小的 scene-private adapter，而不是 full fine-tuning？**  
现有 unseen-environment 方案大多依赖 target fine-tuning、层选择迁移或 meta-learned initialization；这些方法虽然有效，但解释方式仍然是“目标域来了以后把原模型再调一遍”。TUNE 已经做了“动态保留哪几层”；Cross-Environment TL 已经做了“少量目标样本 fine-tune 到 target gNB”；MAML 已经做了“few-shot 快速更新”；AMBER 与 JCN 2025 等则进一步把 backbone 做得更复杂。对 Globecom/ICC 投稿来说，如果你再走 full fine-tuning 路线，只会得到“又一个 adaptation trick”的评价。本文应该把问题重新写成：**shared branch 已经学到跨场景共享规律，因此目标域只需更新 scene-private adapter 与少量 prototype/normalization 参数，即可恢复 target-specific 细粒度映射。** 这比 full fine-tuning 更节省参数，更容易复现，也更有物理可解释性。citeturn5view4turn6view0turn5view5turn12view0turn6view3

## 完整方法框架设计

我建议把整篇论文的方法统一命名为一条主线：**Shared-Private Knowledge Decoupling + Hierarchical Beam Semantics + Lightweight Target Adapter + Coarse-Prototype Alignment**。这里最重要的不是模块越多越好，而是每个模块都服务于同一个论点：**跨场景要迁移 coarse semantics，不要强行迁移 fine mapping；新地点只更新 scene-private correction。**

在 backbone 上，不建议再做很重的 BEV fusion 或大尺寸 transformer。基于 challenge 输入格式与现有最强方法的复杂度，建议采用一个**可复现的轻量模态编码 + 轻量 token fusion** 框架：RGB 用 MobileNetV3-Small 或 EfficientNet-B0 级别编码器，输入下采样到 224×224；LiDAR 不走重型 point transformer，而采用 range-view/front-view 投影后的轻量 2D CNN；Radar 先构造 range-angle map，再用 ResNet10/18-lite；GPS 只用两帧位置、相对位移、速度方向与到 BS 的相对极坐标做 2–3 层 MLP；如果想加入历史 beam index，只把它作为**可选输入**，不作为主版本依赖，以免被质疑和 beam tracking 文献混淆。时间建模上，不必堆大时序 transformer：每个模态先做**浅层 temporal conv 或单层 GRU**，再把模态 token 喂给一个 **2-layer、hidden size 256 的 tiny fusion transformer 或 gated token mixer**。这样做的理由很简单：现有文献已经反复证明多模态融合有效，但也已经把“更复杂的融合器”做得很多了；你的论文不应把创新点押在 fusion 器上。citeturn5view1turn5view3turn12view0turn6view3turn41view0

形式化地，设第 \(m\) 个模态的输入序列为 \(\mathbf{x}^{(m)}_{1:T_m}\)，模态编码器为 \(f_m\)，时序聚合器为 \(t_m\)。则模态级时序特征为
\[
\mathbf{z}^{(m)} = t_m\big(f_m(\mathbf{x}^{(m)}_{1:T_m})\big), \quad m \in \{\text{RGB, LiDAR, Radar, GPS, BeamHist}\}.
\]
将所有模态特征投影到统一维度后，得到融合表示
\[
\mathbf{h} = F\big(\{\mathbf{W}_m \mathbf{z}^{(m)} + \mathbf{e}_m\}_{m}\big),
\]
其中 \(F\) 是轻量 token fusion 模块，\(\mathbf{e}_m\) 是模态类型嵌入。然后做 shared/private 分解：
\[
\mathbf{c}=E_c(\mathbf{h}),\qquad \mathbf{s}=E_s(\mathbf{h}),
\]
其中 \(\mathbf{c}\) 承担跨场景可共享的 coarse semantics，\(\mathbf{s}\) 承担 scene-private geometry/style/fine correction。推荐维度配置为 \(d_h=256,\ d_c=128,\ d_s=64\)，故意让 private branch 比 shared branch 更窄，避免 private 分支“吞掉全部信息”。

beam label 不建议继续用单层 64 类 one-hot 直接分类。更合理的是构造**层次化 beam semantics**。设 codebook 大小为 \(B\)（challenge 中 \(B=64\)），每组包含 \(K_0\) 个相邻 beam，则 coarse group 数为 \(G=B/K_0\)。对 beam index \(b\in\{0,\dots,B-1\}\)，定义
\[
g = \left\lfloor \frac{b}{K_0}\right\rfloor,\qquad
\delta = b - gK_0.
\]
coarse head 只看 \(\mathbf{c}\)：
\[
p(g|\mathbf{c}) = C_g(\mathbf{c});
\]
fine head 看 \([\mathbf{c},\mathbf{s},\mathrm{Emb}(g)]\)：
\[
p(\delta|\mathbf{c},\mathbf{s},g) = C_\delta([\mathbf{c};\mathbf{s};\mathrm{Emb}(g)]).
\]
最终 beam 分布可写成
\[
p(b)=p(g)\,p(\delta|g),\qquad b=gK_0+\delta.
\]
如果你想再增加一点物理味道，可以把相邻 beam 的 steering angle 当作顺序先验，在 fine head 上使用**circular label smoothing / angular-aware loss**，而不是普通 CE。

关于 \(K_0\) 的选择，我会直接建议：**主设定用 \(K_0=8\)**，并以 \(K_0\in\{2,4,8,16\}\) 做消融。原因是非常明确的。\(K_0=2\) 时 coarse group 太细，shared branch 仍容易背 scene-specific 细映射，迁移性弱；\(K_0=4\) 会比 \(2\) 稳，但 coarse semantics 仍不够抽象；\(K_0=16\) 又太粗，会把太多判别压力推给 private refinement，使 Top-1 容易掉、目标域适配负担加重；\(K_0=8\) 相当于把 64 beams 压成 8 个 coarse sectors，通常最符合“可共享粗语义 + 需适配精细偏移”的分工。预计它对 Top-3/Top-5/DBA 的收益会比对 Top-1 更稳定，但如果 private adapter 设计得好，Top-1 不应明显受损。更一般地，若 codebook 大小变化，建议让 coarse group 数 \(G\) 保持在 **8–16** 之间，而不是固定 \(K_0\) 数值。

在解耦约束上，建议主损失写成
\[
\mathcal{L} = \mathcal{L}_{\text{hier}} + \lambda_{\text{orth}}\mathcal{L}_{\text{orth}}
+ \lambda_{\text{hsic}}\mathcal{L}_{\text{hsic}}
+ \lambda_{\text{scene}}\mathcal{L}_{\text{scene-priv}}
+ \lambda_{\text{mi}}\mathcal{L}_{\text{mi}}
+ \lambda_{\text{rec}}\mathcal{L}_{\text{rec}}
+ \lambda_{\text{ang}}\mathcal{L}_{\text{ang}}
+ \lambda_{\text{con}}\mathcal{L}_{\text{supcon}}
+ \lambda_{\text{proto}}\mathcal{L}_{\text{proto}}.
\]
其中层次监督损失
\[
\mathcal{L}_{\text{hier}}
= \mathrm{CE}(g^\*, p(g|\mathbf{c})) + \lambda_\delta\,\mathrm{CE}(\delta^\*, p(\delta|\mathbf{c},\mathbf{s},g^\*)).
\]
正交约束用于减少 shared/private 冗余：
\[
\mathcal{L}_{\text{orth}}
= \left\|\frac{\mathbf{C}^\top \mathbf{S}}{N}\right\|_F^2.
\]
为了让 \(\mathbf{c}\) 尽量和 scene label 无关，可以在 source 端最小化
\[
\mathcal{L}_{\text{hsic}} = \mathrm{HSIC}(\mathbf{c}, y_{\text{scene}}),
\]
而为了让 \(\mathbf{s}\) 保留 scene-specific 信息，则额外给 \(\mathbf{s}\) 接一个 scene classifier：
\[
\mathcal{L}_{\text{scene-priv}} = \mathrm{CE}(D_{\text{scene}}(\mathbf{s}), y_{\text{scene}}).
\]
这组设计比 DANN 更合理，因为 DANN 对整块特征施加 domain confusion，容易连 fine beam refinement 所需的 site-specific 信息一起压平；而你这里明确规定：**只有 \(\mathbf{c}\) 要“与场景无关”，\(\mathbf{s}\) 反而必须“与场景有关”。**

互信息最小化可以选一个可实现而不过重的版本，例如用 variational upper bound/CLUB 形式去压低 \(I(\mathbf{c};\mathbf{s})\)。但从半年落地角度，我建议把它作为**可选项**：主文里保留，主实验只做一版，若训练不稳就把它降级成 appendix ablation。重构损失不要回到原始模态重建，那会显著加大系统复杂度；更稳的做法是重构 fused feature：
\[
\hat{\mathbf{h}} = D([\mathbf{c};\mathbf{s}]),\qquad
\mathcal{L}_{\text{rec}} = \|\hat{\mathbf{h}}-\mathbf{h}\|_2^2,
\]
其作用只是防止 shared/private 退化到“一个分支空掉”。beam-aware smoothness 则建议直接围绕 beam 邻域构造软标签：
\[
q_j \propto \exp\Big(-\frac{d_{\text{circ}}(j,b^\*)^2}{2\sigma^2}\Big),\qquad
\mathcal{L}_{\text{ang}} = -\sum_j q_j \log p(j),
\]
这样相邻 beam 被错预测时不会被像普通 CE 那样“一票否决”，更符合 challenge DBA 的物理语义。监督对比损失则只放在 \(\mathbf{c}\) 上，以同 coarse group 为正样本，不同 coarse group 为负样本，用来增强 coarse semantics 的聚类性。相关文献已充分说明 Top-k 之外的物理指标、LOS/NLOS 区分和 coarse 邻域语义在 beam 任务中的重要性，因此这样的层次化/平滑化设计是有明确动机的。citeturn45view0turn33view1turn33view0

为了避免“private branch 偷偷学完所有信息，shared branch 失效”，我建议你在实现上再加三条很关键的机制。第一，**coarse head 只接 \(\mathbf{c}\)**，绝不让 \(\mathbf{s}\) 直接参与 coarse prediction。第二，给 private 分支更小容量，比如 \(d_s=64\) 而 \(d_c=128\)，并在 source 训练时对 \(\mathbf{s}\) 加 dropout，让 coarse 任务不能依赖 private 特征。第三，在实验中加入**泄漏检测 probe**：用线性 probe 从 \(\mathbf{c}\) 预测 scene label、从 \(\mathbf{s}\) 预测 coarse group。理想现象是“scene-from-\(\mathbf{c}\)”接近随机，“group-from-\(\mathbf{s}\)”明显弱于“group-from-\(\mathbf{c}\)”。这会极大增强你论文里“不是名义上拆分”的可信度。

在 target adaptation 阶段，建议采用**source-free + 小参数 adapter** 路线。冻结所有模态 encoder、时间模块、fusion 模块、shared encoder 与 coarse head，只更新：private branch 末端的 adapter、fine head 的最后一层、少量 normalization 参数，以及 target prototype bank。推荐的 adapter 形式是最朴素也最稳的 bottleneck residual adapter：
\[
A_t(\mathbf{s}) = \mathbf{s} + \mathbf{U}\,\sigma(\mathbf{V}\mathbf{s}),
\]
其中 \(\mathbf{V}\in\mathbb{R}^{r\times d_s}\)、\(\mathbf{U}\in\mathbb{R}^{d_s\times r}\)，\(r\ll d_s\)。如果你更喜欢 LoRA，也只把 LoRA 加在 \(E_s\) 和 \(C_\delta\) 上，不要碰 shared branch。合理的可训练参数比例是**0.5\%–2\%**，再高就会被质疑“本质还是 fine-tuning”。这一设计与 TUNE/MAML/Cross-Environment TL 的核心差别也就很清楚了：**他们在学更新模型；你在学更新 scene-private correction。**citeturn5view4turn5view5turn6view0

针对三种 target 设置，可以统一成一个框架。无标签适配时，使用冻结的 source coarse prototypes \(\{\boldsymbol{\mu}_g\}_{g=1}^G\) 和 evolving target-private prototypes \(\{\boldsymbol{\nu}_g\}_{g=1}^G\)。source coarse prototype 从 source 训练集的 shared 特征均值构造：
\[
\boldsymbol{\mu}_g = \frac{1}{|\mathcal{D}_g|}\sum_{i:y_i^g=g}\frac{\mathbf{c}_i}{\|\mathbf{c}_i\|_2}.
\]
对 target 样本 \(i\)，先算 soft assignment
\[
\alpha_{ig} = \frac{\exp(\mathrm{sim}(\mathbf{c}_i,\boldsymbol{\mu}_g)/\tau)}
{\sum_{g'}\exp(\mathrm{sim}(\mathbf{c}_i,\boldsymbol{\mu}_{g'})/\tau)}.
\]
再用高置信样本更新 target-private prototype
\[
\boldsymbol{\nu}_g \leftarrow m\boldsymbol{\nu}_g + (1-m)\,\overline{\mathbf{s}'_i},
\quad \mathbf{s}'_i=A_t(\mathbf{s}_i),\quad
\text{if }\max_g \alpha_{ig}>\eta_g.
\]
无标签目标函数可写成
\[
\mathcal{L}_{\text{unsup}}
= \mathcal{L}_{\text{IM}}
+ \lambda_{p}\sum_i\sum_g \alpha_{ig}\|\mathbf{c}_i-\boldsymbol{\mu}_g\|_2^2
+ \lambda_{tp}\sum_i\sum_g \alpha_{ig}\|\mathbf{s}'_i-\boldsymbol{\nu}_g\|_2^2
+ \lambda_{cons}\mathcal{L}_{\text{cons}},
\]
其中 \(\mathcal{L}_{\text{IM}}\) 采用经典的“逐样本低熵 + 批量高多样性”的 information maximization 形式，以防所有样本塌缩到一个 coarse group。极少标签 setting 下，只需在上式上加入少量 target labeled samples 的 hierarchical CE，同时允许标注样本直接锚定 \(\boldsymbol{\nu}_g\)。半监督 setting 下，再加 teacher-EMA 伪标签与强弱增强一致性即可。

这个 prototype alignment 设计的关键是：**只在 coarse semantic space 对齐到 source，而不是直接在 absolute beam class space 做对齐。** 这就是你与 generic prototype alignment 最大的区别，也是与现有 beam prediction 文献差异最大的地方。为减轻 confirmation bias，我建议使用三条保险机制：先做 2–3 epoch 的 coarse-only warm-up，不立刻更新 fine offset 伪标签；只使用 coarse confidence 高、且弱增强/模态子集增强下预测一致的样本做 prototype update；prototype 更新采用 class-balanced EMA，保证热门 group 不会压死稀有 group。整个算法可以用如下伪代码概括：

```text
Algorithm: SKD-HBSA Training

Input:
  Source labeled set Ds, target set Dt
  Frozen lightweight multimodal backbone
Output:
  Source model + target private adapter At + target prototypes {νg}

Stage A: Source pretraining
  for each source batch:
      encode modalities -> h
      split h -> c, s
      predict coarse group g and fine offset δ
      update source coarse prototypes {μg} by EMA on c
      optimize Lhier + Lorth + Lhsic + Lscene-priv + Lrec + Lang (+ optional Lsupcon)

Stage B: Target adaptation
  freeze modality encoders, fusion, Ec, coarse head
  initialize At and target-private prototypes {νg}
  for each target batch:
      encode -> h -> c, s
      s' = At(s)
      compute group prediction p(g|c)
      compute soft assignment α to source coarse prototypes {μg}
      if confidence > threshold:
          update νg by EMA using s'
      optimize:
          unlabeled: LIM + Lproto + Lcons
          few-shot:  + Lhier on tiny labeled target set
          semi-supervised: + pseudo-label CE + consistency
```

如果你最终要冲击 Globecom/ICC，我建议主文只保留最有力的四个技术关键词：**hierarchical beam semantics、shared-private decoupling、lightweight target adapter、coarse-prototype alignment**。MI、contrastive、复杂 teacher-student 都可以作为可选增强，不要把论文写成 loss 拼装。

## 实验协议与评价指标

实验协议一定要比方法更“审稿友好”。DeepSense6G challenge 本身就是为了研究 unseen scenario generalization 而设，scenarios 31–34 都属于 street-level V2I beam prediction 数据；31 与 32 为 daytime，33 与 34 为 nighttime。与此同时，挑战评估采用 DBA score，公开后续分析工作也指出 beam prediction 不能只看 accuracy，尤其 GPS 质量会对多模态预测造成强烈影响，LOS/NLOS 与 beam 邻域语义也会改变指标解释。因此，你的实验协议必须避免两个 reviewer 雷区：**伪 target leakage** 与 **把 mixed-shift 误写成 same-location-only shift**。citeturn17search3turn17search1turn19search0turn20search3turn45view0turn33view1turn33view0

我建议主实验分成四组。第一组是**严格 leave-one-scene-out new-location split**：在 \(\{31,32,33,34\}\) 上做 4 折，每次留 1 个场景为 target，其余 3 个作 source。比如 \(\{31,32,33\}\to34\)、\(\{31,32,34\}\to33\)、\(\{31,33,34\}\to32\)、\(\{32,33,34\}\to31\)。这组实验最重要，因为它最贴合 challenge 的“unseen scenario”精神，也最适合验证你的核心 claim：**shared branch 是否真的学到了可转移 coarse semantics。**

第二组是**day/night style-shift split**，但这里必须谨慎命名。就目前公开可见的网页摘要，能确认的是 31/32 daytime、33/34 nighttime；**不能仅凭公开摘要就武断声称 31↔33 或 32↔34 是同一路段同地点**。因此，论文里最安全的写法有两种。若你后续核实场景页元数据确实表明存在同地点昼夜对，则报告 “same-location different-time” 配对实验；若不能核实，就明确写成 **day/night style-shift split**：\(\{31,32\}\to\{33,34\}\) 与反向 \(\{33,34\}\to\{31,32\}\)。这不是文字游戏，而是避免 reviewer 认为你“错误标注 domain shift 类型”。这一点非常关键。citeturn17search1turn19search0turn20search3

第三组是**few-shot target adaptation split**。对每个 held-out target scene，从 target 中抽取 label budget \(=0/5/10/20/50\) 的有标注样本，其余 target 样本全作为 unlabeled target。这里建议做 5 个随机种子，并优先采用**coarse-group-stratified sampling**，而不是完全随机抽样。原因很直接：如果 10 个样本全落在少数几个 beam neighborhood，few-shot 实验会高度偶然，审稿人会质疑标注预算的公平性。你可以在正文明确说明：当某些 coarse group 在 target 中样本不足时，才退化为 scene-level uniform random。这样 protocol 更严谨，也与你的 hierarchical label 设计一致。

第四组是**cross-modality robustness split**，但它在你这里是副故事，不是主故事。建议包含四种缺失模态 setting：drop camera、drop radar、drop LiDAR、drop GPS；以及四种退化 setting：GPS drift/jump/freeze、相机 blur/over-exposure/partial occlusion、LiDAR sparsification、Radar ghost peak/noise injection。要证明的不是“我比 AMBER 更擅长缺失模态”，而是**你的 knowledge split 在模态退化时仍然能保住 coarse semantics，从而让 adaptation 不至于崩掉**。citeturn6view3turn12view5

评价指标不要只放 accuracy。推荐正文主表统一报告：Top-1、Top-3、Top-5、DBA、normalized received power / power ratio、beam power loss（dB）、adaptation time、trainable parameter ratio、性能-标注预算曲线、性能-适配时间曲线、source-to-target generalization gap、robustness under corruption。它们各自对应的论文 claim 其实很明确。Top-1 证明精确 beam 命中；Top-3/Top-5 证明 candidate reduction 能力；DBA 证明你捕捉到了角度邻域语义而不是单纯分类；power ratio/received power/beam loss 证明通信性能意义；adaptation time 与 trainable parameter ratio 证明“轻量 target adaptation”不是口号；budget curve 证明 few-shot 能力；generalization gap 证明 shared branch 的跨场景价值；corruption robustness 则证明 coarse semantics 没有被个别模态绑架。官方 challenge 和后续指标分析都表明 DBA 与 power-ratio 更贴近 beam 任务的物理目标，因此这两个指标必须进主表，而不是只放附录。citeturn38search5turn45view0turn33view1

## 基线与消融设计

baseline 不能只是“挑几个能跑的”。你这篇论文如果想过 Globecom/ICC，最重要的是让 reviewer 一眼看出：**我知道自己在和哪一类工作比较，并且比较是公平的。** 为此，baseline 应分三组组织。第一组是**无 adaptation 或弱 adaptation**：Source-only、Target-only few-shot、Linear probing、AdaBN/BN-stat alignment。第二组是**经典 adaptation**：Full fine-tuning、DANN、MAML、TUNE-style transfer、Cross-Environment TL-style fine-tuning。第三组是**结构拆解 baseline**：普通 multimodal transformer、普通 shared encoder + classifier、prototype-only adaptation、adapter-only adaptation、你的 full method。相关已有工作的适配策略在文献中都有明确原型，因此无需照搬原文 backbone，而应在**相同轻量 backbone** 上复现它们的“策略”，这是最公平也最容易自证的方法。citeturn5view4turn5view5turn6view0turn25view0turn5view3

公平比较原则建议在论文中写得非常具体。所有 baseline 使用相同数据划分、相同 source scenes、相同输入模态集合、相同数据增强、相同 optimizer 预算；few-shot 比较时 target label budget 完全一致；无标签 target adaptation 只允许使用 unlabeled target，不允许额外 source replay；full fine-tuning 与 adapter-based 方法要同时报告**参数更新比例**与**实际适配时间**；若某 baseline 原文依赖不同 backbone，则在你的实验里固定 backbone，仅保留其核心 adaptation 机制。这样 reviewer 即使不认同你的方法，也很难质疑 protocol。

下面这张消融表足够支撑 Globecom/ICC 主文。重点不是做很多 variant，而是每个 variant 都紧扣一个 claim。

|Variant|Hierarchical label|Shared-private|Orthogonality|HSIC|MI minimization|Target adapter|Prototype alignment|Trainable params|Expected conclusion|
|---|---|---|---|---|---|---|---|---|---|
|V0 Plain MM backbone|否|否|否|否|否|否|否|100%|纯 fusion 基线，作为所有收益参照|
|V1 + Hierarchical label|是|否|否|否|否|否|否|100%|证明 coarse/fine label 比 flat 64-class 更适合 DBA 与 Top-3|
|V2 + Shared-private|是|是|否|否|否|否|否|100%|证明“显式知识分工”本身就优于单表示|
|V3 + Orthogonality|是|是|是|否|否|否|否|100%|证明仅拆分不够，需减少冗余泄漏|
|V4 + HSIC|是|是|是|是|否|否|否|100%|证明 shared 分支确实更 scene-invariant|
|V5 + MI min|是|是|是|是|是|否|否|100%|验证更强解耦是否进一步有益，若收益小可降为附录|
|V6 Adapter-only|是|是|是|是|否|是|否|约1%–2%|证明轻量 adapter 已能替代 full FT 的大部分收益|
|V7 Prototype-only|是|是|是|是|否|否|是|约0.1%–0.3%|证明 coarse semantic alignment 对 0-label target 有价值|
|V8 Full method|是|是|是|是|可选|是|是|约1%–2%|证明 adapter 与 prototype 是互补而非可替代|
|V9 Full FT upper-bound|是|是|是|是|可选|否|否|100%|作为“性能上限/代价上限”参照，不是主方向|

每个消融想证明的逻辑是连贯的。V1 证明**为什么层次语义是更好的 transfer unit**；V2–V4 证明**为什么必须显式 shared/private 而不是靠网络自己学**；V6–V8 证明**为什么目标域只更新 private adapter + prototype 就够了**；V9 则帮助你在 rebuttal 中回答“为什么不直接 full fine-tune”。若 full FT 的收益只比 V8 高 0.5–1.0 个点，但代价大 50–100 倍参数更新和若干倍适配时间，那你的故事就立住了。

## 创新性评估与投稿风险

以审稿人视角看，这个方向**是足够新的，但只有在你把创新点描述对了的前提下**。如果摘要写成“we propose a domain adaptation framework for multimodal beam prediction”，新意会直接掉到中低；如果写成“we propose a lightweight target adapter for few-shot beam prediction”，也会和 TUNE/Cross-Environment TL/MAML 形成明显重叠。真正能让 reviewer 眼前一亮的表述，应当是：**we study what knowledge transfers across scenarios in multimodal beam prediction, and show that coarse beam semantics should be shared while scene geometry and fine index mapping should be adapted with a tiny scene-private adapter.** 这个表述与 challenge 论文提出的 unseen-scenario 问题、与现实中 mmWave 传播的 site-specific 特性、与 DBA 指标反映的 beam 邻域语义，是逻辑相连的。citeturn44view0turn6view0turn45view0

最容易被质疑重复的对象，一是 TUNE/Cross-Environment TL/DANN-style transfer，二是 Meta-learning unseen environment，三是一切“更强融合 backbone”工作。换句话说，**你的创新点绝不能表述为“更好的 transfer learning”、也不能表述为“更好的 multimodal fusion”**。从风险控制角度，我会建议你把论文中最核心的创新点固定成三句话。第一句：**首次把 beam prediction 的 transferable knowledge 显式拆成 coarse beam semantics 与 scene-private fine refinement。** 第二句：**首次把 prototype alignment 限定在 coarse semantic space，而不是 absolute beam class space。** 第三句：**提出只更新极小 scene-private adapter 的 source-free/few-shot adaptation pipeline，而不是做 full fine-tuning。** 这样就能与现有文献形成清晰边界。citeturn5view4turn5view5turn25view0turn6view3turn46search2

方案里最容易显得像“拼装”的模块，是 MI minimization、contrastive loss、scene classifier、prototype alignment、adapter 同时上阵。如果你把它们平铺直叙地全部写进 main method，审稿人会说“this is a bag of tricks”。因此必须把故事线收紧成一个统一框架：**因为 absolute beam class 不可直接迁移，所以先抽 coarse semantics；因为 coarse semantics 与 scene-private geometry 混在一起会伤害迁移，所以做 shared/private decoupling；因为 target 只需要恢复 scene-private mapping，所以只更新 tiny private adapter；因为 target 无标签时需要语义锚点，所以只在 coarse semantic space 做 prototype alignment。** 只要四个模块按这个因果链讲，loss 再多也不会显得散。

如果投稿 Globecom / ICC，我建议**删减三样东西**。第一，删掉大模型/VLM/LLM 叙事，不要碰。第二，把 MI/contrastive 这种“可有效但不核心”的损失移到附录或补充材料。第三，不要把 missing modality 做成主实验，只保留 robustness 小节。相反，如果未来想往 TVT / TWC 走，可以再加强三样内容：其一，增加一个关于“为何 coarse beam semantics 更可迁移”的理论分析，哪怕只是基于 beam 邻域和 power-ratio/DBA 的 reasoning；其二，增加更系统的 K0 消融、scene-leakage probe 与 source-free/few-shot/semi-supervised 三套完整协议；其三，增加复杂度、适配时间、参数比例和鲁棒性曲线。已有分析型工作已经指出 DeepSense 上单纯靠 GPS 或单一指标很容易得出误导结论，因此更全面的 protocol 与 metric 会显著提升说服力。citeturn33view1turn33view0

下面这段 contribution 可以直接写进 Introduction，并且表述上尽量“具体、可验证、避免空泛”：

第一，我们重新定义了多模态波束预测中的跨场景迁移问题，不再把它视为统一的 domain-invariant representation learning，而是提出**coarse beam semantics 可共享、scene-specific geometry 与 fine beam refinement 必须适配**的知识分工视角。该分工由层次化 beam label、scene leakage probe 和跨场景 DBA / power-ratio 增益共同验证。  
第二，我们提出一个**shared-private knowledge decoupling** 框架，使 shared branch 只学习 coarse beam semantics，private branch 只承担场景相关的几何与风格修正；与 DANN-style 全表示对齐不同，该设计具有更清晰的物理解释，并在 unseen target scenes 上显著缩小 source-to-target generalization gap。  
第三，我们提出一个**lightweight target adaptation** 机制，在 source-free、few-shot 和 semi-supervised 三种 target 设置下，只更新约 1% 量级的 scene-private adapter，并利用**coarse-prototype alignment** 对 target shared semantics 进行无标签对齐，从而避免 full fine-tuning 的高成本。  
第四，我们在 DeepSense6G scenarios 31–34 上建立严格的 leave-one-scene-out、day/night-style shift、few-shot budget 和 modality corruption 实验协议，并同时报告 Top-k、DBA、power-ratio、adaptation time 与 trainable parameter ratio，全面验证所提方法的泛化性、通信意义与部署效率。citeturn44view0turn45view0turn33view1

## 半年内可落地的最终执行方案

如果目标是**半年内做完、优先冲 Globecom**，我建议你直接把项目压缩成一个“**最小而完整**”的版本，不追求模型炫技，追求 story 一致与结果扎实。

推荐题目可以从下面五个里挑。  
**Shared and Private Knowledge for Cross-Scenario Multimodal Beam Prediction in New Locations**  
**What to Transfer for Multimodal Beam Prediction in Unseen mmWave V2X Scenes**  
**Hierarchical Beam Semantics and Lightweight Scene-Private Adaptation for Cross-Scenario Beam Prediction**  
**Coarse-to-Fine Knowledge Transfer for Multimodal Beam Prediction in DeepSense6G**  
**Lightweight Source-Free Adaptation for Multimodal Beam Prediction via Shared-Private Beam Semantics**

方法缩写我建议准备三个备选，最终选一个最顺口的：  
**SKIP-Beam**：Shared-private Knowledge decoupling with hIerarchical Prototypes for Beam prediction  
**HiST-Beam**：Hierarchical Semantics Transfer for Beam prediction  
**SPARTA-Beam**：Shared-Private Ada pteR for Target Adaptation in Beam prediction  
如果只选一个，我更推荐 **HiST-Beam**，因为它最贴你论文的主线，不会让 reviewer 误以为你在讲复杂系统工程。

模型结构图的文字描述可以这样写：输入为 RGB、LiDAR、Radar、GPS 的短历史序列；每个模态先经过轻量 encoder 与简短时序聚合；融合模块将模态 token 压成一个统一表示 \(\mathbf{h}\)；\(\mathbf{h}\) 再被分成 shared 表示 \(\mathbf{c}\) 与 private 表示 \(\mathbf{s}\)；shared head 预测 coarse beam group，private-aware head 以 \([\mathbf{c},\mathbf{s},g]\) 为输入做 fine offset refinement；source 阶段用 scene-aware decoupling losses 训练 shared/private 分工并构建 source coarse prototypes；target 阶段冻结绝大部分参数，只更新 private adapter、fine head 与 target-private prototypes。

训练流程建议分两步。第一步是 source pretraining：用 source scenes 训练轻量 backbone、hierarchical head 与 shared/private decoupling，保存 source coarse prototypes。第二步是 target adaptation：加载冻结的 source model，在 target 端根据有无标注分别运行 unsupervised / few-shot / semi-supervised 版本，仅更新 scene-private adapter、fine-head 最后一层、少量 normalization 参数和 target-private prototype bank。

测试流程也保持极简。先跑 source-only zero-shot 得到 baseline；再在 target 数据上依据标注预算执行适配；最终在 target test split 报告 Top-1/3/5、DBA、power-ratio、beam loss、adaptation time 与 trainable params。若担心 reviewer 质疑“few-shot label selection 太偶然”，就固定 5 个随机种子并报告均值与标准差。

关键公式其实只需要保留四组，足以支撑整篇论文。  
第一组是 shared/private 分解：
\[
\mathbf{c}=E_c(\mathbf{h}),\qquad \mathbf{s}=E_s(\mathbf{h}).
\]
第二组是层次 beam 监督：
\[
g=\lfloor b/K_0\rfloor,\qquad \delta=b-gK_0,\qquad
p(b)=p(g)\,p(\delta|g).
\]
第三组是解耦与场景约束：
\[
\mathcal{L}_{\text{dec}}=
\lambda_{\text{orth}}\mathcal{L}_{\text{orth}}
+\lambda_{\text{hsic}}\mathcal{L}_{\text{hsic}}
+\lambda_{\text{scene}}\mathcal{L}_{\text{scene-priv}}.
\]
第四组是 target adaptation 的 prototype alignment：
\[
\alpha_{ig}\propto \exp(\mathrm{sim}(\mathbf{c}_i,\mu_g)/\tau),\qquad
\mathcal{L}_{\text{proto}} =
\sum_{i,g}\alpha_{ig}\|\mathbf{c}_i-\mu_g\|_2^2
+\beta\sum_{i,g}\alpha_{ig}\|\mathbf{s}'_i-\nu_g\|_2^2.
\]

实验表格规划可以直接定成六张主表加三张主图。  
主表一：4-fold LOSO new-location 总结果，列出 Source-only、AdaBN、DANN、TUNE-style、Cross-Env TL-style、Adapter-only、Prototype-only、HiST-Beam。  
主表二：few-shot budget \(0/5/10/20/50\) 曲线表。  
主表三：parameter ratio 与 adaptation time。  
主表四：模态缺失与 corruption robustness。  
主表五：K0 消融与 decoupling 消融。  
主表六：scene leakage probes 与 shared/private 可解释性结果。  
主图一：模型结构图。  
主图二：Performance vs Label Budget。  
主图三：Performance vs Adaptation Time。  
如果版面吃紧，主表六可以挪补充材料。

预期结果趋势可以先在论文计划里定下来。最重要的不是绝对数值，而是相对关系。你应该期待：在 0-label target 上，HiST-Beam 相比 Source-only 在 DBA 上有稳定提升，并且优于 DANN/AdaBN 这类“全表示域对齐”；在 5–20 labels 下，HiST-Beam 会迅速逼近 full fine-tuning，但适配时间和参数更新量远低于 full FT；在 Top-1 上，V8 full method 应明显优于 adapter-only 与 prototype-only，证明两者互补；在 Top-3/DBA 与 power-ratio 上，hierarchical label 的收益会比 flat classifier 更稳；在 day/night 或模态噪声下，shared branch 能保住 coarse group，因此 DBA 降幅小于普通 transformer baseline。基于现有文献的性能区间与 challenge 难度，我会把“**zero-shot target 明显优于 Source-only，few-shot 10–20 labels 达到 full FT 大部分收益**”作为最现实也最有说服力的目标，而不是盲目追求全维度 absolute SOTA。citeturn5view3turn12view0turn38search10turn6view3

代码实现优先级必须非常明确。  
优先级最高的是 **P0：数据划分、DBA/Top-k/power-ratio 评估脚本、结果可复现日志**。没有这个，后面都会乱。  
接着做 **P1：轻量 source-only backbone**，先用 RGB+Radar+GPS 跑通。  
然后是 **P2：hierarchical labels 与 coarse/fine head**。  
再做 **P3：shared-private decoupling + leakage probe**。  
之后才是 **P4：adapter-based target adaptation**。  
最后上 **P5：coarse prototype alignment**。  
LiDAR 建议放在 **P6**：先作为增强版，不要一上来就让整个工程复杂度失控。  
这种优先级排序能保证你在第 4–6 周就拿到一版可投稿雏形。

一个紧凑的 8–12 周路线图，可以按下面执行。  
第 1–2 周：固定 LOSO split、day/night split 命名规范、DBA/power-ratio 实现；复现 Source-only 与官方 baseline。  
第 3–4 周：完成轻量 backbone 与 hierarchical label；出第一张主结果表。  
第 5–6 周：加入 shared/private 分解、orthogonality、scene classifier、HSIC；跑第一轮消融。  
第 7–8 周：实现 adapter-only 与 prototype-only，再合并成 full method；完成 0/5/10/20/50 few-shot 实验。  
第 9–10 周：补 DANN、TUNE-style、Cross-Environment TL-style、公平对比；补 trainable params 与 adaptation time。  
第 11–12 周：补 corruption robustness、画图、清理论文故事、删掉收益小且训练不稳的额外 loss。  
如果进度紧张，第 11–12 周就只补最必要的 robustness 与 K0 消融，先投 Globecom。

最小可发表版本我会这样定义：**RGB+Radar+GPS 三模态；K0=8；shared-private + orthogonality + scene-private supervision + coarse prototype alignment + bottleneck adapter；四折 LOSO + few-shot 0/10/50；baseline 至少包含 Source-only、AdaBN、DANN、TUNE-style、Full FT；指标至少包含 Top-1/Top-3/DBA/power-ratio/adaptation time/params ratio。** 这已经足够构成一篇结构完整且创新点清晰的 Globecom/ICC 论文。

强化版期刊版本则可以在六个方向上展开：加入 LiDAR 全模态版本；把无标签、few-shot、半监督三种 adaptation 都做全；加入泄漏 probe 与更多解释性分析；加入更系统的 K0 与 prototype 设计消融；增加 modality corruption/open-world robustness；补一个关于 coarse semantics 可迁移性的理论讨论。这样到 TVT/TWC 层面就更有厚度。

最终一句话总结你的最佳落点：**不要把这篇论文写成“如何微调 target 场景”，而要写成“在多模态毫米波波束预测里，究竟什么知识应该迁移，什么知识应该只由一个极小的 scene-private adapter 去承担”。** 只要这个主线不偏，这个方向完全有机会在半年内做出一篇不会与 TUNE / MAML / DANN / AMBER / 一般 multimodal transformer 正面重合、且审稿人能看懂其增量价值的 Globecom / ICC 论文。