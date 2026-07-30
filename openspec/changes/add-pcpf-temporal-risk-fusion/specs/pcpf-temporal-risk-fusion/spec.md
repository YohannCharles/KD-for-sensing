## ADDED Requirements

### Requirement: PCPF-T 必须保持四模态输入与研究隔离

系统 MUST 以 `image、radar、gps、lidar` canonical order 接受五帧历史并预测一个 64 类未来 beam。模型、loss 和风险 target MUST NOT 读取 channel、CSI、path、beam power、历史 beam index、天气、场景、domain、corruption type 或 severity；风险估计前 MUST NOT 执行跨模态 attention 或 feature concat。全部当前开发结果 MUST 标记 `claim_ineligible=true`，outer test MUST 保持未访问。

#### Scenario: 构建 PCPF-T batch
- **WHEN** runner 从合法 MMW train/validation batch 构建模型输入
- **THEN** 输入 MUST 只包含四模态历史 tensor、`modality_temporal_mask` 与未来 beam label
- **AND** 模型 forward MUST 不接受 CSI/channel/天气/场景作为动态风险输入

#### Scenario: 请求 outer test
- **WHEN** PCPF-T 配置未获得新的显式 outer-test 授权
- **THEN** runner MUST NOT 构建 outer-test loader
- **AND** 输出 metadata MUST 保持 `claim_ineligible=true` 与 `outer_test_accessed=false`

### Requirement: 共享 Temporal Transformer 必须正确屏蔽缺失帧

系统 MUST 将 encoder 输出校验为 `[B,5,64]`，stack 为 `[B,5,4,64]`，对每模态应用独立 input LayerNorm/可选 adapter，再以 `[B*4,5,64]` 送入唯一共享的两层、四头、`dim_feedforward=128`、`norm_first=true`、非 causal Transformer。系统 MUST 使用 learned time embedding `[5,64]`、modality embedding `[4,64]` 和共享 T-CLS；`src_key_padding_mask` MUST 让 T-CLS 可见并屏蔽全部缺失 frame。

#### Scenario: 部分帧缺失
- **WHEN** `modality_temporal_mask` 为 `[B,5,4]` 且某些 frame cell 为 false
- **THEN** false cell MUST 不作为 attention key/value 参与 temporal encoding
- **AND** 输出 MUST 包含 `temporal_token_features=[B,5,4,64]`、`temporal_cls_features=[B,4,64]` 与 `temporal_attention_valid_fraction=[B,4]`

#### Scenario: 整个模态缺失
- **WHEN** 一个样本的某模态五帧全部为 false
- **THEN** 该模态 CLS/frame feature MUST 在 Transformer 后显式置零
- **AND** `available_modalities`、probability、risk 与 weight MUST 对该模态置为 false/零

### Requirement: 四个专家必须共享唯一 Beam Prototype Bank

系统 MUST 只实例化一个 `BeamPrototypeBank`，其 64 个 `[64]` prototype 同时为全部可用模态产生 cosine/temperature logits。Stage 1 MUST 输出 `unimodal_logits=[B,4,64]` 和 probability，并复用现有 topology soft target、availability-aware fused/modality prototype alignment。新增 unimodal loss MUST 对每样本可用模态的 hard CE 与 soft topology CE 求和后按可用模态数归一化。

#### Scenario: 同一样本有两个模态可用
- **WHEN** Stage 1 计算该样本的 unimodal loss
- **THEN** 两个模态 MUST 查询同一个 prototype Parameter
- **AND** hard/soft loss MUST 除以二且 unavailable 模态 MUST 不贡献 loss

#### Scenario: 没有逐帧监督标签
- **WHEN** batch 只提供一个未来 beam label
- **THEN** 系统 MUST NOT 将该 label 复制为五个 frame-level supervised target

### Requirement: Stage 1 必须只训练 temporal experts 与 prototype

`stage1_expert` MUST 训练四个 encoder、encoder projection/adapter、共享 Temporal Transformer、Beam Prototype Bank 和当前 deterministic prediction component。默认融合 MUST 为所有可用模态 uniform probability average；可选 static learnable prior control MUST 只训练四个全局 prior logits。probability/risk head、dynamic analytic fusion、direct Router control 和 U0 Router oracle loss MUST 冻结或不存在。

#### Scenario: 默认 Stage 1 backward
- **WHEN** 对 Stage 1 loss 执行一次 backward
- **THEN** expert/temporal/prototype 中参与 loss 的参数 MUST 获得有限梯度
- **AND** probability head、risk coefficient、temperature、tau 与 direct Router MUST 没有梯度

### Requirement: Stage 2 概率嵌入必须可校准且 eval 确定

系统 MUST 计算 `mu=h+DeltaMu(h)` 与 clamped `logvar=LogVarHead(h)`；DeltaMu 末层 MUST 零初始化，LogVarHead 初始输出 MUST 约为 -4，默认 clamp MUST 为 `[-8,4]`。训练 MAY 以重参数采样的 `z` 计算 preserve loss，但 `R_star`、risk、fused evaluation MUST 使用 deterministic `mu` probability；eval 重复 forward MUST 一致。

#### Scenario: 新建 Stage 2 模型
- **WHEN** probability head 尚未训练
- **THEN** `mu` MUST 与输入 CLS 相等到浮点容差
- **AND** logvar MUST 落在配置 clamp 范围内且均值接近 -4

#### Scenario: eval 重复执行
- **WHEN** 相同输入在 `model.eval()` 下 forward 两次
- **THEN** unimodal probability、risk、weight 与 fused probability MUST 逐值一致

### Requirement: topology risk 必须由四个受限分量生成

系统 MUST 计算 `U_var=mean(exp(logvar))`、`U_proto=1-max cosine(mu,C)`、temporal circular trajectory residual `U_temp` 和与其他可用模态均值分布的 JS conflict `U_conflict`。只有至少三帧有效时才能拟合 temporal 一阶趋势；不足三帧 MUST 返回 `U_temp=0,temp_valid=false`。Single 模态时 MUST 返回 `U_conflict=0`。风险 MUST 为 `softplus(sum softplus(rho_x)*normalize(U_x)+bias)`，默认系数跨模态共享且非负；不得使用任意 MLP 或 modality identity 预测 risk。

#### Scenario: circular trajectory 跨越 63 到 0
- **WHEN** frame prototype distribution 的 circular mean 从 label 63 平滑移动到 label 0
- **THEN** 展开和线性残差 MUST 使用最短 circular difference
- **AND** 不得产生接近整圈的虚假跳变

#### Scenario: 仅一个模态可用
- **WHEN** availability 行只有一个 true
- **THEN** 所有模态 conflict MUST 为零
- **AND** unavailable 模态的四项风险与 raw risk MUST 为零

### Requirement: 风险监督和拟合状态必须只来自 train split

四项 normalization mean/std 与 Stage 3 的 `mean_train_risk_m` MUST 只遍历 train dataset 拟合并作为 buffer 冻结；validation、historical development test 与 outer test MUST 不更新它们。`R_star_m` MUST 使用 detached deterministic unimodal probability 与当前 topology 的 normalized circular distance，Dmax MUST 来自 topology，且 unavailable 模态 MUST 排除。

风险分量 empirical std 低于预注册 `0.01` normalization std floor 时 MUST 保存并使用 `0.01`，不得以 `1e-6` 机器精度 epsilon 缩放可训练风险分量。

Stage 2 loss MUST 为 masked Huber risk loss、只对 `|R_star_a-R_star_b|>rank_margin` 激活的 pair ranking、小权重 Gaussian KL 和保持 beam 语义的 topology preserve CE；MUST NOT 使用 fused beam CE 推动风险值。

#### Scenario: 拟合 risk normalization
- **WHEN** Stage 2 从 Stage 1 validation-best 初始化
- **THEN** preparation MUST 只读取 train dataset 并保存 split identity/count/mean/std
- **AND** validation forward MUST 不改变任何 normalization buffer

#### Scenario: 构造风险目标
- **WHEN** 对 `R_star` 求 loss 并 backward
- **THEN** `R_star` 使用的 probability MUST detach
- **AND** 风险 target 路径 MUST 不向 expert/prototype 或 unimodal logits 反传梯度

#### Scenario: 初始 U_var 退化为常数
- **WHEN** Stage 2 preparation 在初始恒定 logvar 上拟合得到 `U_var std=0`
- **THEN** checkpoint 与 preparation report 中保存的 `U_var std` MUST 至少为 `0.01`
- **AND** 随后一个 Stage 2 优化步 MUST 保持梯度有限且可用模态 raw risk 不得全部变成精确零

### Requirement: Stage 2 gate 必须在 Stage 3 前失败关闭

系统 MUST 从 Stage 2 validation-best 生成 overall、每模态、sunny/rainy/foggy、15 domain、Full/drop-1/drop-2/Single 的 Pearson/Spearman、calibration/decile、top/bottom 20% 与 confident-but-wrong 报告。tracked config MUST 预注册 overall Spearman `>0.20`、至少三模态正相关、每种天气 overall 正相关、最高 20% 真实风险高于最低 20% 及无常数化门槛。

#### Scenario: 任一 gate 失败
- **WHEN** evaluator 发现一个预注册条件不满足
- **THEN** 报告 MUST 写 `stage2_gate_passed=false` 和具体原因
- **AND** launcher MUST NOT 自动启动或 resolve 可训练的 Stage 3 配置

#### Scenario: gate 通过
- **WHEN** 所有预注册条件满足
- **THEN** Stage 3 resolved config MUST 绑定 gate JSON 与 SHA256
- **AND** 阈值 MUST 与 tracked config 相同且不得由评估器回写

### Requirement: Stage 3 必须使用固定解析式概率融合

系统 MUST 从 train-only `mean_train_risk_m` 计算 `a_m=exp(-eta*mean_train_risk_m)`，以 `T_m=Tmin+softplus(t_m)` 校准每模态 probability，并以 `tau=tau_min+softplus(raw_tau)` 计算 `score_m=availability_m*a_m*exp(-risk_m/tau)` 与归一化 weight。实现 MUST 在 FP32 log-score 中等价计算以防下溢。最终 `p_fused` MUST 为校准单模态 probability 的 weighted sum，不得添加 MLP 权重修正。

#### Scenario: missing 与 Single 权重
- **WHEN** 任意非空 availability mask 进入 Stage 3
- **THEN** missing weight MUST 严格为零且每行 weight 和 MUST 为一
- **AND** Single mask 的唯一可用模态 weight MUST 严格为一

#### Scenario: Stage 3A backward
- **WHEN** analytic fusion NLL 执行 backward
- **THEN** 默认只有四个 temperature 与 tau 获得梯度，可选 eta 仅在显式启用时获得梯度
- **AND** expert、prototype、probability/risk head MUST 保持冻结

### Requirement: 四个训练 stage 必须记录精确冻结与 checkpoint 角色

系统 MUST 只接受 `stage1_expert`、`stage2_risk`、`stage3_fusion`、`stage3b_optional_finetune`，启动时 MUST 输出完整 trainable parameter names/count 并断言不存在额外可训练参数。checkpoint payload/metadata MUST 记录 stage、fusion mode、claim eligibility 和 fitted-state identity；Stage 2/3/3B MUST 分别拒绝非 Stage 1/2/3A validation-best 初始化。现有 U0 checkpoint 缺少新 metadata 时 MUST 继续按原路径加载。

#### Scenario: stage 来源不匹配
- **WHEN** Stage 3 配置指向 metadata.stage=`stage1_expert` 的 checkpoint
- **THEN** initialization MUST 在 optimizer 或训练 step 前失败

#### Scenario: 加载旧 U0 checkpoint
- **WHEN** U0 配置没有声明 PCPF source-stage 要求
- **THEN** checkpoint loader MUST 不要求 PCPF metadata
- **AND** U0 state dict 与 forward MUST 继续严格加载

### Requirement: 对照和消融必须共享同一专家证据

系统 MUST 提供 Uniform、Static Prior、Direct Router control、CUAF-style `local_adaptation` 和 PCPF-T analytic mode，以及 no-var/no-proto/no-temp/no-conflict/no-static-prior/no-risk-supervision。A0--A4 MUST 绑定同一 Stage 1 checkpoint fingerprint、split、seed、optimizer budget 与 validation-loss selection；dynamic replacement MUST 缓存同一次 forward 的 unimodal logits，不能通过重跑 encoder 制造差异。

#### Scenario: 构建 A4 主模型
- **WHEN** fusion mode 为 PCPF-T analytic
- **THEN**模型 MUST 不含直接输出四维权重的 Router MLP

#### Scenario: 比较已训练 Direct Router
- **WHEN** evaluator 加载 A2 control checkpoint
- **THEN** A2 与 A4 的 Stage 1 expert fingerprint MUST 完全相同
- **AND** 不匹配时比较 MUST 失败而不是继续汇总

#### Scenario: 汇总已训练 A0--A3 control
- **WHEN** evaluator 将 A0--A3 validation-best checkpoint 与 A4 汇总为同一矩阵
- **THEN** control MUST 在同一次 A4 forward 缓存的 unimodal logits 与风险分量上应用各自已训练的 temperature、tau 或 Router 参数
- **AND** evaluator MUST NOT 为 control 重跑 encoder，并 MUST 记录每个 control checkpoint 的路径、SHA256、role、fusion mode 与 expert fingerprint

### Requirement: 评估必须输出性能、校准和机制诊断

系统 MUST 复用 Full Top-1/3/5、Single Macro/Worst、All-14 Macro/Worst、Within-3、circular MAE、四个 Missing、四个 Single、sunny/rainy/foggy 和 15-domain macro/worst。每个 mask/weather MUST 额外输出 weight mean/std/percentile、相对 static prior 偏差、missing weight max、effective modality count、risk-weight Spearman、真实风险排序一致率、NLL/Brier/ECE/reliability data 与 temperature。confident-but-wrong MUST 使用每模态 train-only 90% confidence threshold。

#### Scenario: 运行 validation 15-mask evaluator
- **WHEN** evaluator 使用一个 validation-best checkpoint 和同一 validation split
- **THEN** 所有 15 个非空 mask MUST 使用同一模型参数与单模态 evidence
- **AND** 输出 MUST 包含 claim eligibility、checkpoint、split、weather/domain 和 normalization provenance

### Requirement: 配置、数值和 smoke 必须失败关闭

PCPF parser MUST 拒绝未知字段、负 loss/risk 系数、非正 temperature、`d_model % num_heads != 0`、非五帧 seq、非法 stage、缺少 stage checkpoint/gate 的训练请求。risk/softmax/exp/log/KL MUST 在 FP32 执行，即使主模型为 BF16。实现 MUST 提供 static/focused tests、synthetic forward/backward、真实 MMW 单 batch Stage 1、Stage 1 假 checkpoint的 Stage 2 和 Stage 3 smoke，并报告 shape、loss 分量、关键梯度、missing weight、row-sum、NaN/Inf 与 GPU peak memory。

#### Scenario: BF16 风险 forward
- **WHEN**主 expert tensor 为 BF16
- **THEN** risk component、raw risk、fusion score 与 KL 的内部输出 MUST 为 FP32
- **AND** canonical logits MAY 在稳定计算完成后 cast 回模型 dtype

#### Scenario: 非法配置字段
- **WHEN** PCPF loss/model config 包含未声明字段
- **THEN** parser MUST 在模型训练前列出未知字段并失败
