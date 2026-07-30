你正在修改仓库：

https://github.com/YohannCharles/KD-for-sensing

目标：实现一个新的“多帧拓扑风险校准解析式动态融合”研究分支。该分支不让神经网络直接输出四个模态权重，而是：

1. 用轻量 Temporal Transformer 对每个模态的5帧历史独立建模；
2. 通过共享 Beam Prototype Bank 训练四个单模态专家；
3. 学习每个模态当前样本的连续 beam-topology risk；
4. 使用固定解析公式把风险转换成四个模态权重；
5. 在 sunny/rainy/foggy、任意模态缺失下进行评估。

暂定方法名：

PCPF-T
Prototype-Calibrated Precision Fusion with Temporal Risk

==================================================
一、先审计仓库，不要直接重写
==================================================

先阅读并理解至少以下文件及其调用链：

- src/kd_sensing/models/u_mask_beam_jepa.py
- src/kd_sensing/losses/u_mask_beam_jepa.py
- src/kd_sensing/losses/u_mask_beam_jepa_config.py
- src/kd_sensing/losses/u_mask_beam_jepa_prototype.py
- src/kd_sensing/losses/beam_prototype_alignment.py
- configs/mmw/u0.yaml
- 当前数据 loader、temporal mask、available_modalities 契约
- 当前训练 extension、checkpoint、validation-best 选择逻辑
- 当前15-mask评估与天气/场景分组代码
- 最新 CSI 分支及其边界

输出一份简短审计，说明：

1. encoder 当前输出是否为 [B,T,D]；
2. modality_temporal_mask 的真实形状与语义；
3. BeamPrototypeBank 当前的 logits、prototype 参数和 describe 接口；
4. prototype_alignment_loss 是否已经使用 topology soft target；
5. 当前训练流程如何冻结参数、加载 initialization checkpoint；
6. 新分支应该新增哪些文件，哪些旧文件只复用、不修改。

必须保持：

- 当前 U0 数值兼容；
- 当前 CSI/M4/TSPC 代码兼容；
- 当前 checkpoint 可正常加载；
- 默认配置未启用 PCPF-T 时，不新增参数、不改变 forward；
- 不删除任何旧 Router 代码，它仍作为对照基线。

优先新增独立注册模型，例如：

pcpf_temporal_risk_fusion

不要把新逻辑继续堆进 UMaskBeamJEPA，除非仓库架构明确要求通过 opt-in extension 实现。

==================================================
二、任务边界
==================================================

数据：

- MMW；
- 四模态顺序必须复用仓库 canonical order：
  image、radar、gps、lidar；
- 历史窗口 T=5；
- 预测未来1个64类 beam；
- 仅使用自然 sunny/rainy/foggy；
- 使用现有完整、drop-1、drop-2、drop-3/单模态缺失模式；
- 不合成人工噪声、遮挡、漂移或传感器 corruption。

禁止：

- 不使用 channel、CSI、path、beam-power 作为输入、训练 target 或风险标签；
- 不使用历史 beam index 作为输入；
- 不使用天气标签、场景标签作为 Router/风险模块的推理输入；
- 不访问 outer test；
- 不根据 historical development test 调参；
- 不引入直接输出四维权重的自由 MLP Router；
- 不把 corruption type、severity 等不存在的信息送入模型。

现阶段只使用 frozen inner train/validation 和 historical development test，所有结果标记：

claim_ineligible: true

==================================================
三、模型总体结构
==================================================

每个模态输入：

x_m ∈ [B,5,...]

经过当前对应 encoder：

H_m = Encoder_m(x_m) ∈ [B,5,64]

四个 encoder 保持独立。

之后使用：

模态专属输入适配器
    ↓
共享轻量 Temporal Transformer
    ↓
每模态窗口表示 h_m ∈ [B,64]
    ↓
共享 Beam Prototype Bank
    ↓
每模态 beam logits/probability
    ↓
概率嵌入与风险量
    ↓
固定解析公式生成4个模态权重
    ↓
概率级融合

禁止在风险估计前进行跨模态 attention 或特征拼接。

==================================================
四、Temporal Transformer
==================================================

新增共享 Temporal Transformer，默认配置：

- d_model: 64
- num_layers: 2
- num_heads: 4
- dim_feedforward: 128
- dropout: 0.1
- norm_first: true
- causal: false
- sequence length: 1个T-CLS + 5帧
- learned time embedding: [5,64]
- learned modality embedding: [4,64]
- learned T-CLS token：可以每模态独立，也可以共享token加modality embedding；
  默认优先共享T-CLS，减少参数。

每个模态有独立：

- input LayerNorm；
- 可选 Linear(64,64) adapter。

四个模态共享后面的两层 TransformerEncoder。

建议一次性把张量重排为：

[B,M,T,D] → [B*M,T,D]

加入：

frame_feature
+ time_embedding
+ modality_embedding

然后添加 T-CLS，得到：

[B*M,6,D]

使用 src_key_padding_mask：

- T-CLS 永远可见；
- 缺失时间帧不可被 attention；
- 不允许补零帧参与 attention。

若某个模态5帧全部缺失：

- Transformer可以执行，但最终 h_m 必须显式置零；
- available_m=false；
- 该模态后续 probability、risk、weight 均被 mask；
- 不允许仅凭 T-CLS 或 modality embedding 产生伪模态证据。

输出以下诊断：

- temporal_token_features: [B,5,4,64]
- temporal_cls_features: [B,4,64]
- temporal_attention_valid_fraction: [B,4]
- available_modalities: [B,4]
- temporal_pooling_type: shared_temporal_transformer
- temporal_pooling_param_count

不得使用 causal mask，因为全部5帧都是历史可用观测。

==================================================
五、共享 Beam Prototype Bank
==================================================

复用现有 BeamPrototypeBank，不复制第二套 prototype。

64个 prototype：

C = {c_0,...,c_63}, c_k ∈ R^64

对每个可用模态：

l_m(k) = cosine(h_m,c_k) / temperature

输出：

unimodal_logits: [B,4,64]
unimodal_probabilities: [B,4,64]

共享 prototype 的训练目标是：

- 同一个样本中，所有可用模态表示均向真实 beam prototype 靠近；
- 相邻 beam 根据现有 circular topology soft target 得到较小惩罚；
- 远距离 beam 得到较大惩罚。

先审计现有 prototype_alignment_loss。

如果它已经正确实现：

- fused feature alignment；
- modality feature alignment；
- circular/topology soft label；
- availability mask；

则直接复用，不重复实现另一套 InfoNCE。

新增显式单模态 beam loss：

L_unimodal =
  对每个样本的所有可用模态计算 topology-aware CE，
  再按该样本可用模态数量归一化。

需要同时保留：

- hard label CE；
- topology soft-label CE。

建议：

L_uni_m =
    lambda_hard * CE(logits_m,y)
  + lambda_soft * CE(logits_m,q_y)

默认：

lambda_hard = 1.0
lambda_soft = 0.5

不要默认加入模态—模态直接对比损失。

不同模态通过共享 beam prototype 间接语义对齐，避免强制 image/radar/gps/lidar 的表示逐点相同。

不要使用历史帧 beam label。

如果 batch 没有明确、无泄漏的逐帧标签，就不实现 frame-level supervised CE；配置必须默认关闭，不能从未来标签伪造5个帧级标签。

==================================================
六、三阶段训练
==================================================

实现显式 stage 枚举：

training_stage:
- stage1_expert
- stage2_risk
- stage3_fusion
- stage3b_optional_finetune

每个阶段必须：

- 明确参数冻结列表；
- 启动时打印 trainable parameter names/count；
- fail closed 检查不应训练的参数；
- checkpoint metadata 中记录 stage；
- 禁止把不匹配阶段的 checkpoint 静默加载。

--------------------------------------------------
Stage 1：Temporal Experts + Shared Prototype
--------------------------------------------------

训练：

- 四个 modality encoders；
- encoder projections/adapters；
- shared Temporal Transformer；
- BeamPrototypeBank；
- 必要的 deterministic prediction components。

关闭/冻结：

- Gaussian variance head；
- topology risk calibration；
- dynamic analytic fusion；
- 原有 supervised Router；
- router oracle loss。

Stage 1 不训练动态权重。

融合控制至少实现：

1. uniform：
   所有可用模态等权；
2. static learnable prior control：
   仅使用4个全局 prior logits，经 availability masked softmax。

主模型推荐先使用 uniform，确保专家不是依靠强模态权重掩盖弱专家。

损失：

L_stage1 =
    L_fused_hard
  + lambda_uni * L_unimodal
  + 现有 prototype alignment loss
  + 可选 topology soft-label fused loss

建议初值：

lambda_uni = 1.0
lambda_proto = 沿用当前 U0 的成熟值
lambda_modality_proto = 沿用当前 U0 的成熟值
beam_label_sigma = 2.0

复用项目已验证的 Full/Single 均衡或交替训练策略，不重新设计数据划分。

Stage 1 保存：

stage1_best.pth

checkpoint 只能由 inner validation loss 选择。

Stage 1 必须输出：

- 每模态 Full 条件 Top-1/Top-3/Within-3/MAE；
- 每个 Single 条件成绩；
- prototype pairwise cosine/distance；
- prototype有效秩；
- 每模态到真实 prototype 的平均距离；
- 每模态 logits 的 NLL、Brier、ECE；
- temporal attention/CLS 特征是否出现 NaN或坍缩。

--------------------------------------------------
Stage 2：概率嵌入与拓扑风险学习
--------------------------------------------------

从 stage1_best.pth 初始化。

冻结：

- 所有大 encoder；
- encoder projections；
- shared Temporal Transformer；
- BeamPrototypeBank；
-原有 classifier/Router；
- static prior。

新增每个模态共享或轻量共享的概率头：

mu_m = h_m + DeltaMu(h_m)
logvar_m = clamp(LogVarHead(h_m), min_logvar, max_logvar)

要求：

- DeltaMu末层零初始化，使初始 mu=h；
- LogVarHead初始输出约 -4；
- 默认 logvar clamp 为 [-8,4]；
- train时可重参数采样；
- eval默认使用 mu，保证确定性。

z_m = mu_m + exp(0.5*logvar_m)*epsilon

共享 prototype bank对 mu/z 输出单模态 logits。

构造四个解析风险分量：

1. latent variance

U_var_m = mean(exp(logvar_m))

2. absolute prototype distance

U_proto_m =
    1 - max_k cosine(mu_m,c_k)

必须使用绝对cosine距离，不能只使用softmax entropy。

3. temporal trajectory residual

对每个有效历史帧 token计算 frame prototype distribution。
由64-beam circular topology计算每帧 circular beam mean。

仅当有效帧数 >= 3 时：

- 在圆周展开后拟合一阶趋势 theta_hat(t)=a+bt；
- 计算 circular residual mean square；
- 将结果归一化至稳定范围。

有效帧数 < 3 时：

- U_temp=0；
- 另外输出 temp_valid=false；
- 不允许除零或伪造趋势。

该项只做无标签风险特征，不使用历史 beam label。

4. weak cross-modal conflict

U_conflict_m =
    JS(p_m || mean_{j!=m, available} p_j)

当只有一个模态可用时：

U_conflict=0

该项必须是弱项，不允许单独决定权重。

风险预测使用固定低自由度公式，不使用任意MLP：

raw_risk_m =
    softplus(
        lambda_var * normalize(U_var_m)
      + lambda_proto * normalize(U_proto_m)
      + lambda_temp * normalize(U_temp_m)
      + lambda_conflict * normalize(U_conflict_m)
      + bias
    )

约束：

lambda_x = softplus(rho_x) >= 0

默认所有 lambda 在四模态间共享。

不要把 modality one-hot、天气、domain、availability pattern 输入动态风险公式。

各风险分量归一化统计必须只用 training split 拟合并冻结；
validation/test不得更新。

构造训练风险真值：

R_star_m =
    sum_k p_m_detached(k)
          * normalized_circular_distance(k,y)

要求：

- p_m_detached必须detach；
- y为真实未来beam；
- 距离使用项目已有active topology；
- Dmax根据拓扑正确计算；
- unavailable modality不参与；
- 不读取beam power。

损失：

L_risk =
  masked Huber(raw_risk,R_star)

L_rank =
  对同一样本的可用模态两两比较；
  仅当 |R_star_a-R_star_b| > rank_margin 时激活；
  保证真实低风险模态预测风险也更低。

L_kl =
  Gaussian KL，使用很小beta，防止方差无限漂移。

L_preserve =
  sampled z 或 mu 的单模态 topology CE，
  防止概率头破坏 Stage 1 beam语义。

L_stage2 =
    lambda_risk * L_risk
  + lambda_rank * L_rank
  + beta_kl * L_kl
  + lambda_preserve * L_preserve

建议初值：

lambda_risk = 1.0
lambda_rank = 0.2
rank_margin = 0.05
beta_kl = 1e-4
lambda_preserve = 0.2

Stage 2 不训练最终动态融合权重，也不使用 fused beam CE推动风险值投机。

Stage 2 保存：

stage2_best.pth

晋级Stage 3前必须生成风险可观测性报告：

- overall Spearman/Pearson(raw_risk,R_star)
- 每个模态
- sunny/rainy/foggy
- 15个domain
- Full/drop-1/drop-2/Single
- calibration curve
- risk decile vs empirical topology error
- confident-but-wrong子集：
  单模态confidence高但Top-1错误时，risk是否显著高于正确样本

预注册最低门槛先写入配置，不在跑完后修改：

- overall Spearman > 0.20
- 至少3/4模态 Spearman > 0
- 每种天气 overall Spearman > 0
- risk最高20%样本的真实风险高于最低20%
- 不出现方差/风险常数化

若门槛失败：

- 自动标记 stage2_gate_passed=false；
- 不自动启动Stage 3；
- 如实输出失败原因。

--------------------------------------------------
Stage 3：解析式动态融合
--------------------------------------------------

从 stage2_best.pth 初始化。

冻结：

- encoder；
- Temporal Transformer；
- BeamPrototypeBank；
- probability/risk heads，Stage 3A先全部冻结。

先计算train-only静态模态能力：

mean_train_risk_m =
  训练集该模态所有可用样本R_star的平均值

a_m =
  exp(-eta * mean_train_risk_m)

归一化后保存到checkpoint和JSON。

不得使用validation/test拟合a_m。

每模态概率校准：

p_cal_m =
  softmax(unimodal_logits_m / T_m)

约束：

T_m = Tmin + softplus(t_m)
默认 Tmin=0.05

动态权重固定为：

score_m =
    availability_m
    * a_m
    * exp(-risk_m / tau)

w_m =
    score_m / sum_j score_j

tau = tau_min + softplus(raw_tau)

默认 tau_min=0.05。

要求：

- missing modality weight严格为0；
- 仅有一个模态时该模态weight严格为1；
- 所有样本权重和为1；
- 不允许另外添加MLP修正权重；
- 不允许使用模态identity进入dynamic residual；
- 长期模态差异只由a_m表达。

概率融合：

p_fused =
  sum_m w_m * p_cal_m

训练使用：

NLL = -log(p_fused[y])

可增加 topology soft-label CE。

Stage 3A只训练：

- T_image/T_radar/T_gps/T_lidar；
- tau；
- 可选eta；
- 若明确配置，可对4个lambda进行极小范围微调；
  默认lambda冻结。

建议5—10 epoch。

Stage 3B仅在Stage 3A超过static prior后启用：

- 解冻概率/risk heads；
- encoder、Temporal Transformer、Prototype仍冻结；
- 学习率为Stage 2的0.05—0.1；
- 2—5 epoch；
- 必须同时监控risk Spearman，若明显下降则停止并回退Stage3A。

==================================================
七、对照与消融
==================================================

在相同 Stage 1 expert checkpoint 上实现以下融合替换，避免不同backbone造成不公平：

A0 Uniform
- 可用模态等权。

A1 Static Prior
- 只使用a_m，不使用样本级risk。

A2 Direct Router Control
- 使用旧式 reliability/entropy/margin/confidence/logit norm MLP直接输出权重；
- 仅作为历史负结果复现；
- 不允许复用不同专家checkpoint。

A3 CUAF-style Analytic Control
- 只使用 entropy、prediction margin、cross-modal divergence；
- 根据论文公式实现；
- 若无法与原论文公式逐项一致，名称必须标记 local_adaptation，不宣称精确复现。

A4 PCPF-T Full
- a_m × exp(-risk/tau)。

消融：

A4-no-var
A4-no-proto
A4-no-temp
A4-no-conflict
A4-no-static-prior
A4-no-risk-supervision

所有方法：

- 使用相同Stage 1初始化；
- 相同split、seed、epoch、optimizer；
- checkpoint均由validation loss选择；
- 不允许从test选择方法。

==================================================
八、关键评估
==================================================

复用项目当前正式指标：

- Full Top-1/Top-3/Top-5
- Single Macro/Worst
- All-14 Macro/Worst
- Within-3
- circular beam-index MAE
- Missing Image
- Missing Radar
- Missing GPS
- Missing LiDAR
- image_only及其他Single
- sunny/rainy/foggy
- 15 domain macro/worst

若当前development evaluation本来合法提供normalized gain，可仅用于评估；
不得进入训练target或风险公式。

必须增加：

1. 动态替换实验

在同一个checkpoint、同一批unimodal logits下比较：

- dynamic analytic weights
- static prior weights
- uniform weights
- old direct Router weights

这是判断动态样本级风险是否真正有价值的核心。

2. 权重诊断

对每个mask和天气输出：

- mean weight
- sample weight std
- weight percentiles
- mean absolute dynamic deviation from static prior
- missing weight max
- effective number of modalities
- risk/weight Spearman
- 单模态真实风险排序与权重排序一致率

3. confident-but-wrong诊断

定义每模态：

confidence >= 该模态训练集90%分位
且单模态Top-1错误

统计：

- old Router平均权重
- static prior权重
- PCPF-T权重
- U_var/U_proto/U_temp/U_conflict
- 最终融合是否纠正错误

4. 校准指标

- NLL
- Brier score
- ECE
- reliability diagram数据
- 每模态温度T_m

==================================================
九、工程实现要求
==================================================

建议新增文件，但以审计后的仓库架构为准：

- src/kd_sensing/models/temporal_transformer.py
- src/kd_sensing/models/pcpf_temporal_risk.py
- src/kd_sensing/losses/pcpf_temporal_risk.py
- src/kd_sensing/losses/pcpf_temporal_risk_config.py
- scripts/run_pcpf_stage1.py
- scripts/run_pcpf_stage2.py
- scripts/run_pcpf_stage3.py
- scripts/eval_pcpf.py
- configs/mmw/pcpf/stage1.yaml
- configs/mmw/pcpf/stage2.yaml
- configs/mmw/pcpf/stage3.yaml
- configs/mmw/pcpf/ablations/*.yaml
- tests/test_pcpf_temporal_transformer.py
- tests/test_pcpf_risk.py
- tests/test_pcpf_fusion.py
- tests/test_pcpf_stage_freezing.py
- tests/test_pcpf_evaluator.py

必须复用：

- registries；
- optimizer/scheduler factory；
- training extension；
- checkpoint schema；
- validation-best规则；
- current MMW split；
- current temporal missing schedule；
- current metrics utilities。

禁止复制整套trainer。

配置解析必须：

- 拒绝未知字段；
- 检查lambda非负、temperature正数；
- 检查d_model可被num_heads整除；
- 检查seq_length与数据一致；
- 检查stage和checkpoint role一致；
- 在未声明PCPF时不实例化新参数。

所有公式中的risk/softmax/exp/log/KL：

- 强制float32计算；
- 即使主训练使用bf16；
- 防止exp(-risk/tau)下溢；
- 最后再cast回模型dtype。

==================================================
十、单元测试
==================================================

至少覆盖：

1. [B,5,4,64] → [B,4,64] shape正确；
2. Transformer共享参数而非4份复制；
3. 时间位置和modality embedding shape正确；
4. 缺失帧不参与attention；
5. 整模态缺失时latent/risk/weight为0或masked；
6. 单模态可用时weight=1；
7. missing weight严格为0；
8. 每行weight和为1；
9. 解析融合没有自由四维Router MLP；
10. risk系数均非负；
11. R_star使用detach的概率；
12. circular topology边界0/63正确；
13. temporal residual跨0/63不会产生虚假大跳变；
14. Stage1只训练expert/prototype；
15. Stage2只训练probability/risk heads；
16. Stage3A只训练温度与融合标量；
17. train-only normalization/prior不读取validation/test；
18. eval模式确定性；
19. bf16主模型下风险公式仍为float32；
20. 旧U0测试和CSI相关测试全部继续通过。

==================================================
十一、运行策略
==================================================

本轮先完成实现与预检，不自动启动长训练。

依次执行：

1. 静态检查；
2. focused unit tests；
3. 旧U0兼容测试；
4. synthetic forward/backward；
5. 真实MMW一个batch的Stage1 smoke；
6. 从Stage1假checkpoint进行Stage2 smoke；
7. Stage3解析融合smoke；
8. 生成resolved configs和launch scripts。

smoke必须打印：

- 每阶段trainable parameters；
- 输入输出shape；
- loss各分量；
- prototype梯度；
- risk head梯度；
- temperature/tau梯度；
- missing weight max；
- weight row-sum error；
- NaN/Inf检查；
- GPU显存峰值。

不要启动outer test，不要自动运行多seed或整晚实验。

==================================================
十二、最终交付
==================================================

完成后返回：

1. 修改/新增文件列表；
2. 当前架构到PCPF-T架构的数据流；
3. Stage1/2/3每阶段冻结与训练参数表；
4. 所有损失公式及代码位置；
5. 所有测试命令与结果；
6. smoke结果；
7. resolved config路径；
8. launcher命令；
9. 尚未运行的长实验清单；
10. 风险与已知限制。

特别说明：

- 不要把“代码可运行”写成“方法有效”；
- Stage2 risk gate未通过前，不宣称动态融合成立；
- A4未稳定超过同checkpoint的A1 Static Prior前，不把PCPF-T升级为第二创新点；
- 不因结果不理想而事后修改晋级阈值；
- 保留所有负结果和机制诊断。