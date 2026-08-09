# PCPF Temporal Risk Fusion Specification

## Purpose

定义隔离的 PCPF-T temporal expert、共享 beam prototype、概率风险、三阶段 checkpoint、历史 sparse CSI 扩展与有限 RF beam probing 诊断契约。

## Requirements

### Requirement: PCPF-T 默认四模态且可显式追加历史 sparse CSI

系统 MUST 默认以 `image、radar、gps、lidar` canonical order 接受五帧历史并预测一个 64 类未来 beam。只有配置显式声明 `use_sparse_csi=true` 时，系统 MAY 在末尾追加同一五帧历史窗口的固定 sparse CSI 作为第五模态；默认基线 MUST 使用 2x2 selection，只有预注册 C2 筛选 MAY 使用 4x2 selection。模型、loss 和风险 target MUST NOT 读取当前/未来 CSI、未来 channel、path、beam power、历史 beam index、天气、场景、domain、corruption type 或 severity；风险估计前 MUST NOT 执行跨模态 attention 或 feature concat。全部当前开发结果 MUST 标记 `claim_ineligible=true`，MMW test MUST 默认封存。

#### Scenario: 构建 PCPF-T batch
- **WHEN** runner 从合法 MMW train/validation batch 构建模型输入
- **THEN** 默认输入 MUST 只包含四模态历史 tensor、`modality_temporal_mask` 与未来 beam label
- **AND** 未显式启用 sparse CSI 时模型 MUST 不创建 CSI 参数或改变四模态 state dict/forward

#### Scenario: 请求 test
- **WHEN** PCPF-T 开发配置请求 test role，或未显式授权 test evaluation
- **THEN** runner MUST 在 dataset 创建前拒绝
- **AND** 输出 metadata MUST 保持 `claim_ineligible=true` 与 `outer_test_accessed=false`

### Requirement: 共享 Temporal Transformer 必须正确屏蔽缺失帧

系统 MUST 将每个 encoder 输出校验为 `[B,5,64]`，stack 为 `[B,5,M,64]`，其中默认 `M=4`、sparse CSI opt-in 时 `M=5`。系统 MUST 对每模态应用独立 input LayerNorm/可选 adapter，再以 `[B*M,5,64]` 送入唯一共享的两层、四头、`dim_feedforward=128`、`norm_first=true`、非 causal Transformer。系统 MUST 使用 learned time embedding `[5,64]`、modality embedding `[M,64]` 和共享 T-CLS；`src_key_padding_mask` MUST 让 T-CLS 可见并屏蔽全部缺失 frame。

#### Scenario: 部分帧缺失
- **WHEN** `modality_temporal_mask` 为 `[B,5,4]` 且某些 frame cell 为 false
- **THEN** false cell MUST 不作为 attention key/value 参与 temporal encoding
- **AND** 输出 MUST 包含 `temporal_token_features=[B,5,M,64]`、`temporal_cls_features=[B,M,64]` 与 `temporal_attention_valid_fraction=[B,M]`

#### Scenario: 整个模态缺失
- **WHEN** 一个样本的某模态五帧全部为 false
- **THEN** 该模态 CLS/frame feature MUST 在 Transformer 后显式置零
- **AND** `available_modalities`、probability、risk 与 weight MUST 对该模态置为 false/零

### Requirement: 所有专家必须共享唯一 Beam Prototype Bank

系统 MUST 只实例化一个 `BeamPrototypeBank`，其 64 个 `[64]` prototype 同时为全部可用模态产生 cosine/temperature logits。Stage 1 MUST 输出 `unimodal_logits=[B,M,64]` 和 probability，并复用现有 topology soft target、availability-aware fused/modality prototype alignment。新增 unimodal loss MUST 对每样本可用模态的 hard CE 与 soft topology CE 求和后按可用模态数归一化。

#### Scenario: 同一样本有两个模态可用
- **WHEN** Stage 1 计算该样本的 unimodal loss
- **THEN** 两个模态 MUST 查询同一个 prototype Parameter
- **AND** hard/soft loss MUST 除以二且 unavailable 模态 MUST 不贡献 loss

#### Scenario: 没有逐帧监督标签
- **WHEN** batch 只提供一个未来 beam label
- **THEN** 系统 MUST NOT 将该 label 复制为五个 frame-level supervised target

### Requirement: Stage 1 必须只训练 temporal experts 与 prototype

`stage1_expert` MUST 训练全部已启用 encoder、encoder projection/adapter、共享 Temporal Transformer、Beam Prototype Bank 和当前 deterministic prediction component。默认融合 MUST 为所有可用模态 uniform probability average；可选 static learnable prior control MUST 只训练 `M` 个全局 prior logits。probability/risk head、dynamic analytic fusion、direct Router control 和 U0 Router oracle loss MUST 冻结或不存在。

#### Scenario: 默认 Stage 1 backward
- **WHEN** 对 Stage 1 loss 执行一次 backward
- **THEN** expert/temporal/prototype 中参与 loss 的参数 MUST 获得有限梯度
- **AND** probability head、risk coefficient、temperature、tau 与 direct Router MUST 没有梯度

### Requirement: Stage 2 原型条件 evidential probability 必须保持专家均值

系统 MUST 从冻结的 Stage 1 prototype logits 计算 `q_m=softmax(l_m)`，共享 evidence head MUST 只预测正标量 concentration `kappa_m`，并定义 `alpha_m=kappa_m*q_m`。Dirichlet expectation MUST 等于 Stage 1 的 `q_m`；Stage 2 MUST NOT 计算或添加 `DeltaMu`、改变 prototype logits、采样 feature embedding，或使用 per-dimension Gaussian variance。训练和评估 MUST 使用同一确定性 expert probability。

#### Scenario: 新建 Stage 2 模型
- **WHEN** evidence head 尚未训练
- **THEN** concentration MUST 为配置的正初值
- **AND** `alpha/alpha.sum(-1)` MUST 与 Stage 1 `q_m` 相等到浮点容差

#### Scenario: Stage 2 更新 evidence
- **WHEN** Stage 2 完成任意次优化并与来源 Stage 1 在相同输入上比较
- **THEN** unimodal logits 与 Top-1 MUST 逐值不变
- **AND** 只有 concentration、risk coefficient 与 risk bias MAY 改变

#### Scenario: eval 重复执行
- **WHEN** 相同输入在 `model.eval()` 下 forward 两次
- **THEN** unimodal probability、risk、weight 与 fused probability MUST 逐值一致

### Requirement: topology risk 必须由四个受限分量生成

系统 MUST 计算 `U_concentration=C/(C+kappa)`、prototype probability 在预测 beam 拓扑半径外的质量 `U_neighbor`、temporal circular trajectory residual `U_temp` 和当前模态与其他可用模态均值分布之间的期望 circular topology distance `U_conflict`。只有至少三帧有效时才能拟合 temporal 一阶趋势；不足三帧 MUST 返回 `U_temp=0,temp_valid=false`。Single 模态时 MUST 返回 `U_conflict=0`。风险 MUST 为 `softplus(sum softplus(rho_x)*normalize(U_x)+bias)`，默认系数跨模态共享且非负；不得使用任意 MLP、modality identity、天气或场景预测 risk。

#### Scenario: circular trajectory 跨越 63 到 0
- **WHEN** frame prototype distribution 的 circular mean 从 label 63 平滑移动到 label 0
- **THEN** 展开和线性残差 MUST 使用最短 circular difference
- **AND** 不得产生接近整圈的虚假跳变

#### Scenario: 仅一个模态可用
- **WHEN** availability 行只有一个 true
- **THEN** 所有模态 conflict MUST 为零
- **AND** unavailable 模态的四项风险与 raw risk MUST 为零

### Requirement: 风险监督和拟合状态必须只来自 train split

四项 normalization mean/std 与 Stage 3 的 `mean_train_risk_m` MUST 只遍历 train dataset 拟合并作为 buffer 冻结；validation/test MUST 不更新它们。`R_star_m` MUST 使用 detached deterministic unimodal probability 与当前 topology 的 normalized circular distance，Dmax MUST 来自 topology，且 unavailable 模态 MUST 排除。

风险分量 empirical std 低于预注册 `0.01` normalization std floor 时 MUST 保存并使用 `0.01`，不得以 `1e-6` 机器精度 epsilon 缩放可训练风险分量。

Stage 2 loss MUST 为 masked Huber risk loss、只对 `|R_star_a-R_star_b|>rank_margin` 激活的 pair ranking，以及 `U_concentration` 对 `R_star` 的 masked Huber calibration；MUST NOT 使用 fused beam CE、Gaussian KL、sampled preserve CE、PRE 或 exact-class SupCon 推动风险值。

#### Scenario: 拟合 risk normalization
- **WHEN** Stage 2 从 Stage 1 validation-best 初始化
- **THEN** preparation MUST 只读取 train dataset 并保存 split identity/count/mean/std
- **AND** validation forward MUST 不改变任何 normalization buffer

#### Scenario: 构造风险目标
- **WHEN** 对 `R_star` 求 loss 并 backward
- **THEN** `R_star` 使用的 probability MUST detach
- **AND** 风险 target 路径 MUST 不向 expert/prototype 或 unimodal logits 反传梯度

#### Scenario: 初始 U_concentration 退化为常数
- **WHEN** Stage 2 preparation 在初始恒定 concentration 上拟合得到 `U_concentration std=0`
- **THEN** checkpoint 与 preparation report 中保存的 `U_concentration std` MUST 至少为 `0.01`
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

系统 MUST 从 train-only `mean_train_risk_m` 计算静态能力 `a_m=exp(-eta*mean_train_risk_m)`，以 `T_m=Tmin+softplus(t_m)` 校准每模态 probability，并以 `tau=tau_min+softplus(raw_tau)` 和固定 `gamma=max_log_adjustment` 计算 `delta_m=gamma*tanh((risk_m-mean_available(risk))/tau)`。weight MUST 为 availability-masked `softmax(log(a_m)-delta_m)`；任意两个可用模态的动态 log-odds 改变量 MUST 不超过 `2*gamma`。最终 `p_fused` MUST 为校准单模态 probability 的 weighted sum，不得添加 MLP 权重修正。

#### Scenario: missing 与 Single 权重
- **WHEN** 任意非空 availability mask 进入 Stage 3
- **THEN** missing weight MUST 严格为零且每行 weight 和 MUST 为一
- **AND** Single mask 的唯一可用模态 weight MUST 严格为一

#### Scenario: Stage 3A backward
- **WHEN** analytic fusion NLL 执行 backward
- **THEN** 默认只有 `M` 个 temperature 与 tau 获得梯度，可选 eta 仅在显式启用时获得梯度
- **AND** expert、prototype、probability/risk head MUST 保持冻结

### Requirement: 三个训练 stage 必须记录精确冻结与 checkpoint 角色

系统 MUST 只接受 `stage1_expert`、`stage2_risk`、`stage3_fusion`，启动时 MUST 输出完整 trainable parameter names/count 并断言不存在额外可训练参数。checkpoint payload/metadata MUST 记录 stage、fusion mode、claim eligibility 和 fitted-state identity；Stage 2/3 MUST 分别拒绝非 Stage 1/2 validation-best 初始化。

#### Scenario: stage 来源不匹配
- **WHEN** Stage 3 配置指向 metadata.stage=`stage1_expert` 的 checkpoint
- **THEN** initialization MUST 在 optimizer 或训练 step 前失败

### Requirement: 显式三阶段续跑必须失败关闭

系统 MUST 提供本地显式续跑动作，能够等待已启动的 Stage 1 正常完成，再依次运行 Stage 2、无界 Stage 2 gate 与 Stage 3。续跑 MUST 复用现有 resolver、共享 trainer 和 gate evaluator，并让训练阶段与 gate 分别运行于独立进程。每次进入下一步前 MUST 校验上一训练 run 状态为 `complete`、`last.pth` 已达到配置 epoch、stage-specific validation-best checkpoint 完整发布且训练 stage 匹配。各阶段 MUST 保持同一 protocol、audit、seed、物理 batch、worker 和 output lineage；outer test MUST 保持未访问。

#### Scenario: 等待当前 Stage 1 后自动续跑
- **WHEN** 用户显式对一个仍在运行的合法 Stage 1 resolved config 启动续跑动作
- **THEN** 续跑器 MUST 等待该 run 正常完成并验证 Stage 1 validation-best，随后解析并运行 Stage 2
- **AND** MUST 从 Stage 2 validation-best 运行无 batch 上限的只读 gate
- **AND** 只有 gate 通过后才可绑定 gate JSON/SHA256、解析并运行 Stage 3

#### Scenario: 任一前置步骤失败
- **WHEN** run 状态失败或 stale、训练子进程非零退出、epoch 未完成、checkpoint publication/stage 不匹配或 Stage 2 gate 不通过
- **THEN** 续跑器 MUST 立即非零退出并保留错误原因
- **AND** MUST NOT 解析或启动依赖该失败步骤的下一阶段

### Requirement: 对照和消融必须共享同一专家证据

系统 MUST 在同一 checkpoint forward 上提供 Uniform、Static Prior 与 PCPF-T analytic 三种替换概率。三者 MUST 绑定同一 expert fingerprint、split、seed 和 validation identity，并复用同一次 forward 的 unimodal logits，不能通过重跑 encoder 制造差异。

#### Scenario: 构建 A4 主模型
- **WHEN** fusion mode 为 PCPF-T analytic
- **THEN**模型 MUST 不含直接输出四维权重的 Router MLP

#### Scenario: 汇总保留的解析对照
- **WHEN** evaluator 生成 15 或 31-mask matrix
- **THEN** Uniform、Static Prior 与 PCPF analytic MUST 共享同一次 unimodal probability 和 availability evidence
- **AND** evaluator MUST NOT 加载额外 control checkpoint 或重跑 encoder

### Requirement: 评估必须输出性能、校准和机制诊断

系统 MUST 复用 Full Top-1/3/5、Single Macro/Worst、全部非 Full mask Macro/Worst、Within-3、circular MAE、每模态 Missing/Single、sunny/rainy/foggy 和 15-domain macro/worst。默认四模态 MUST 评估 15 个非空 mask，sparse CSI opt-in MUST 评估五模态全部 31 个非空 mask。每个 mask/weather MUST 额外输出 weight mean/std/percentile、相对 static prior 偏差、missing weight max、effective modality count、risk-weight Spearman、真实风险排序一致率、NLL/Brier/ECE/reliability data 与 temperature。confident-but-wrong MUST 使用每模态 train-only 90% confidence threshold。

#### Scenario: 运行 validation mask evaluator
- **WHEN** evaluator 使用一个 validation-best checkpoint 和同一 validation split
- **THEN** 默认所有 15 个或 opt-in 所有 31 个非空 mask MUST 使用同一模型参数与单模态 evidence
- **AND** 输出 MUST 包含 claim eligibility、checkpoint、split、weather/domain 和 normalization provenance

### Requirement: 配置、数值和 smoke 必须失败关闭

PCPF parser MUST 拒绝未知字段、负 loss/risk 系数、非法 concentration bound、非正 temperature/`max_log_adjustment`、`d_model % num_heads != 0`、非五帧 seq、非法 stage、缺少后续 stage checkpoint/gate 的训练请求。trajectory sparse-CSI Stage 1 MAY 显式 fresh start；若提供初始化 checkpoint，则其训练协议与 normalization MUST 与当前 protocol fingerprint/validation 隔离契约一致。evidence/risk/softmax/exp/log MUST 在 FP32 执行，即使主模型为 BF16。实现 MUST 提供 static/focused tests、synthetic forward/backward、真实 MMW 单 batch Stage 1、Stage 1 假 checkpoint的 Stage 2 和 Stage 3 smoke，并报告 shape、loss 分量、关键梯度、单模态不变性、动态 log-odds 边界、missing weight、row-sum、NaN/Inf 与 GPU peak memory。

#### Scenario: BF16 风险 forward
- **WHEN**主 expert tensor 为 BF16
- **THEN** evidence、risk component、raw risk 与 fusion score 的内部输出 MUST 为 FP32
- **AND** canonical logits MAY 在稳定计算完成后 cast 回模型 dtype

#### Scenario: 非法配置字段
- **WHEN** PCPF loss/model config 包含未声明字段
- **THEN** parser MUST 在模型训练前列出未知字段并失败

### Requirement: 历史 sparse CSI 必须固定、复数且可审计

启用 sparse CSI 时，默认基线 MUST 对每个历史 frame 使用固定 pattern index `[0,1]` 与 frequency index `[0,15]`，得到 `[5,2,2]` complex tensor；mother grid MUST 为 `[5,32,16]`，每帧抽样率 MUST 为 `4/(32*16)=0.78125%`。预注册 C2 MAY 使用固定 pattern index `[0,1,2,3]` 与同一 frequency index `[0,15]`，得到 `[5,4,2]` complex tensor、每帧 8 RE、五帧 40 RE 和 `8/(32*16)=1.5625%` 抽样率。sidecar MUST 只接受这两个 canonical descriptor 的 SHA256，并拒绝其他 selection。正式路线 MUST 绑定当前 `mmw_id_stratified_block_v1` seed manifest 的实际 train/validation windows，selection descriptor、selection SHA256、probe codebook logical/file SHA256、physical frequency offset、历史 frame id、channel path identity、protocol/version、block size、manifest/source/window hash 与 cache identity MUST 写入 resolved config 或 sidecar metadata。训练前 cache scan MUST 只遍历 train/validation 并记录 `test_evaluated=false`，并 MUST 发布由完整 block protocol identity、自身 SHA256、selection SHA256 和原内容寻址 cache key 绑定的 packed `[N,M,2]` complex cache；不同 selection MUST 使用独立 packed bundle/cache manifest，旧 split 或 selection cache MUST 失败且要求重建。正式 dataset MUST 严格命中该 bundle，不得在 worker 中回退到 source channel 计算。正式 resolver 还 MUST 绑定并严格校验 RGB/LiDAR 帧缓存和同一 block protocol 的 GPS coordinate cache。生成路径 MUST 不加入 AWGN、pilot dropout、随机 corruption 或任何 current/future CSI。真实 SNR 不可得时 MUST 记录 `snr_available=false`，不得随机生成或用常数冒充。

#### Scenario: 编码历史 sparse CSI
- **WHEN** batch 提供受支持 selection 对应的 `[B,5,M,2]` complex pilot、`[B,5,M]` pattern id 与两个 frequency position
- **THEN** CSI encoder MUST 保留 real/imag 信息并输出 `[B,5,64]` temporal feature
- **AND** 缺少 SNR MUST 不改变该 feature、logit、risk target 或 fusion weight 的定义

#### Scenario: channel 引用或时间顺序不一致
- **WHEN** 任一 channel 文件 stem 不匹配对应历史 frame id，或最后历史 frame 不早于 target
- **THEN** dataset MUST 在返回样本前失败并报告 sample identity

#### Scenario: 正式缓存绑定不完整
- **WHEN** packed CSI bundle 的 SHA256、protocol fingerprint、selection/codebook/cache spec 不匹配，或任一历史 channel path 不在 bundle 中，或严格 RGB/LiDAR/GPS cache 缺失
- **THEN** resolver 或 dataset MUST 在长训练前失败
- **AND** 不得静默回退到 raw channel、在线图像变换、LiDAR BEV 构建或 GPS 文本解析

#### Scenario: 正式 sparse-CSI 物理 batch
- **WHEN** resolver 未收到显式 batch override 并解析正式 sparse-CSI seed1 模板
- **THEN** train/validation batch size MUST 为 64，worker 数 MUST 为 8
- **AND** batch 64 MUST 在 fresh-start 长训练前通过真实 CUDA 单步显存 smoke

### Requirement: 五模态扩展必须保持缺失语义并 fresh start

未启用 sparse CSI 时，模型 MUST 不创建 CSI encoder/projection 或第五模态参数。五模态 Stage 1 MUST fresh start，不得从四模态 checkpoint 扩展；Stage 2/3 MUST 只接受同一五模态 trajectory protocol 下前一阶段的 validation-best checkpoint。

#### Scenario: sparse CSI 完全缺失
- **WHEN** 某样本五帧 CSI mask 全为 false
- **THEN** CSI input、temporal feature、probability、risk 与 weight MUST 全部为零
- **AND** 四个 sensing 专家的 evidence MUST 与对应四模态输入的结果保持一致到浮点容差

#### Scenario: 训练 31-subset schedule
- **WHEN** 一个 epoch 对五模态样本应用 temporal missing
- **THEN** 全部 31 个非空 bitmask MUST 由 global sample position 与 epoch seed 确定性轮转
- **AND** 各 mask 样本数差 MUST 不超过一，epoch report MUST 写出逐 mask 计数

### Requirement: 正式 topology 与评估证据必须阻止后验换标签

正式 PCPF matrix 与 beam probing MUST 绑定已通过完整 domain、64-beam label、power replay与有限数值检查的 `ula_dft_phase_cycle_v1` audit。resolved config、模型构造、Stage 2/3 initialization、validation-best checkpoint、gate 与 matrix MUST 绑定相同 descriptor SHA256、audit 文件 SHA256、protocol audit、split seed、train/validation identity 和 experiment seed；任一文件缺失、内容漂移或 identity 不匹配 MUST 在训练或报告前失败。`cyclic_index_v1` 运行 MUST 保持 `claim_ineligible=true`，且不得通过修改 config 或报告 metadata 后验升级。复用 observation evidence 时 MUST 额外匹配主 checkpoint SHA256、topology、protocol 与 seed；所有 mask MUST 与 Full mask 使用相同样本和顺序。

#### Scenario: 旧 cyclic checkpoint 搭配 formal config
- **WHEN** initialization 或 evaluator 尝试把 `cyclic_index_v1` checkpoint 加载到声明 `ula_dft_phase_cycle_v1` 的模型
- **THEN** 系统 MUST 在加载 state dict 或写出 formal provenance 前拒绝

#### Scenario: 复用不同 checkpoint 的 observation cache
- **WHEN** `--reuse-evidence` 指向由不同主 checkpoint、control checkpoint、protocol、topology 或 seed 生成的 cache
- **THEN** evaluator MUST 拒绝复用且不得把 cache 重标为当前 checkpoint 证据

#### Scenario: mask 样本配对漂移
- **WHEN** 任一 validation mask 缺少、重排或混入不同于 Full mask 的样本 identity
- **THEN** gate 或 matrix MUST 在聚合指标前拒绝报告

### Requirement: 独立 15/31-mask matrix 必须使用相同 checkpoint 与配对样本

系统 MUST 在同一 `mmw_id_stratified_block_v1` validation identity 上报告冻结 checkpoint 的 15/31-mask prediction，并 MAY 从相同 cached unimodal evidence 计算 uniform、train-only static prior 与当前 analytic PCPF。matrix MUST 绑定同一 split seed、train seed、manifest/window hash、样本顺序、Stage 1/2 fingerprint 和 temperature calibration。旧 split 或 clean-inner 结果不得作为当前初始化或 paired evidence。每个 run MUST 保存样本级 identity/group、label/prediction、每专家 logit/probability/真实 circular error/risk/risk component/weight、availability 与 CSI quality 字段。

#### Scenario: 运行机制诊断
- **WHEN** evaluator 完成五模态 31-mask validation
- **THEN** D0 MUST 使用原始逐样本 dynamic risk，D1 MUST 在相同 domain 与 mask 内确定性打乱 sample risk，D2 MUST 用相同 domain 与 mask 的平均 risk 替换 sample risk，D3 MUST 使用 static prior
- **AND** 四者 MUST 在相同 cached unimodal probability、availability 与样本顺序上重新融合并报告 paired 指标；risk/error、component、confident-wrong、weight transfer 与分布统计 MUST 作为独立机制诊断输出，不得冒充 D0--D3

#### Scenario: 计算不确定性区间
- **WHEN** evaluator 汇总主结果与 paired 差值
- **THEN** bootstrap MUST 以审计通过的独立 group key 重采样，并记录 seed、次数和 group 数
- **AND** 不得把同一 trajectory/window 的帧视为独立样本

### Requirement: 单模态能力诊断必须隔离联合训练与融合

系统 MUST 允许 PCPF sparse-CSI Stage 1 以 `fixed_single_modality` 分别 fresh-start 训练 image、radar、GPS、LiDAR 与 CSI。五条诊断 MUST 绑定相同 `mmw_id_stratified_block_v1` split seed、train seed、训练预算、物理 batch、worker、topology 和 validation-loss checkpoint selection；训练与 validation MUST 始终只开放指定模态。该模式 MUST 禁止 Stage 2/3、关闭 missing-pattern matrix、保持 `claim_ineligible=true`、`test_evaluated=false`，且不得把结果解释为融合提升。

#### Scenario: 训练并验证一个 only 模态
- **WHEN** Stage 1 resolved config 选择一个合法 `fixed_single_modality`
- **THEN** 每个训练 batch MUST 只保留该模态的真实有效历史 cell，逐 epoch validation MUST 使用同一 modality mask
- **AND** 若任一样本的指定模态整段不可用，运行 MUST 失败且不得回退到其他模态

#### Scenario: 在后续阶段请求 only 模态
- **WHEN** resolver 尝试为 Stage 2 或 Stage 3 配置 `fixed_single_modality`
- **THEN** resolver MUST 在生成可训练配置前拒绝

### Requirement: 拓扑原型监督反事实必须保持模型容量一致

系统 MUST 将创新点一限定为 Stage 1 的邻近 beam topology soft CE 与 fused/modality prototype-alignment supervision。无创新点一的反事实 MUST 保留相同的单一 64-beam `BeamPrototypeBank`、共享 prototype logits、fused/unimodal hard CE、模型容量和推理路径，只关闭 topology soft CE 与 prototype-alignment loss；不得以删除 prototype bank、改变 head、禁用 `U_neighbor` 风险分量或复用其他 ablation 冒充该反事实。

#### Scenario: 解析无拓扑原型监督的 Stage 1
- **WHEN** topology-loss-off Stage 1 template 被解析
- **THEN** `unimodal_soft_weight`、`lambda_proto` 与 `lambda_modality_proto` MUST 为零，`use_beam_prototype_alignment` MUST 为 false
- **AND** fused/unimodal hard CE、prototype topology identity、共享 prototype bank、五模态输入、31-subset schedule 与训练预算 MUST 与 topology-loss-on 分支一致

#### Scenario: 对比 topology supervision 开关
- **WHEN** topology-loss-on/off checkpoint 进行 matched validation
- **THEN** 两者 MUST 使用相同 split、预算、head 容量、sample identity、mask 和指标实现，并只按相同 sample/group identity 配对
- **AND** 报告 MUST 明确不同分支不共享 expert fingerprint，且不得引入 Direct Router、CUAF 或动态融合交互项

### Requirement: 已停止的动态融合研究面不得残留 active owner

Direct Router、`cuaf_local_adaptation`、nested-ablation、R0--R7 专用 comparison contract 和逐风险分量 tracked ablation template MUST 从 active model/config/evaluator/test surface 成组移除。当前 checkpoint 所需的 `pcpf_analytic`、uniform/static 基线、风险输出和普通 15/31-mask matrix MUST 保留。历史 analytic run-local config 中已经记录但从未使用的 `direct_router_hidden_dim` MAY 在加载时被丢弃；它 MUST NOT 创建参数、模块、fusion mode、兼容 alias 或运行分支。

#### Scenario: 加载已停止的 fusion mode
- **WHEN** tracked 或用户配置声明 `direct_router_control`、`cuaf_local_adaptation` 或 nested/R0 专用字段
- **THEN** 严格配置解析或 evaluator MUST 拒绝
- **AND** U0、保留 baseline、PCPF analytic checkpoint 和 probing evidence loader MUST 不受影响

#### Scenario: 加载现有 analytic checkpoint provenance
- **WHEN** 历史 run-local analytic config 只额外记录 inert `direct_router_hidden_dim`
- **THEN** loader MAY 丢弃该键并严格加载原 checkpoint
- **AND** model state dict MUST 不含 Direct Router 参数

### Requirement: 弱专家恢复必须先做互斥的 Stage 1 筛选

系统 MUST 将弱专家恢复拆成 J1 联合监督重平衡、C1 CSI token 容量和 R1 Radar 双分支三个互斥 Stage 1 筛选。三个筛选 MUST 绑定相同 `mmw_id_stratified_block_v1` split seed 0、train seed 1、40 epoch、batch 64、8 workers、topology、validation-loss checkpoint selection 与 sealed-test 契约；每条运行 MUST fresh-start、写入独立输出目录并保持 `claim_ineligible=true`。J1 MUST 配置一个除 `lambda_unimodal` 外完全相同的 J0 matched control；第一轮 MUST NOT 同时改变两个模型/目标因素，MUST NOT 启动 Stage 2/3，MUST NOT 根据 validation 结果读取或选择 test。

#### Scenario: 重平衡联合 Stage 1 的专家监督
- **WHEN** J1 五模态联合 Stage 1 template 被解析
- **THEN** `lambda_unimodal` MUST 为 `5.0`，逐样本按可用模态数归一化的 loss 公式、31-subset schedule、hard/soft 比例、prototype supervision 与 encoder 配置 MUST 保持不变
- **AND** J0/J1 MUST 同时使用 BF16 且关闭 GradScaler，二者除 `lambda_unimodal=1.0/5.0` 外 MUST 完全一致；fixed-single 诊断 MUST 将 `lambda_unimodal` 保持为 `1.0`，采用 J1 的 topology on/off 对照 MUST 使用相同的 `lambda_unimodal`

#### Scenario: 筛选固定 20-RE CSI 的 token 容量
- **WHEN** C1 fixed-CSI Stage 1 template 被解析
- **THEN** sparse CSI MUST 仍为五帧 `[5,2,2]` complex observation、固定 pattern/frequency selection 和同一 packed cache，仅 `SparsePilotEncoder.num_layers` MUST 从 0 变为 1
- **AND** 运行 MUST fresh-start，MUST NOT 从零层 CSI encoder checkpoint 迁移或改变 SNR/quality 的 diagnostic-only 边界

#### Scenario: 筛选 Radar RA/DA 双分支 encoder
- **WHEN** R1 fixed-Radar Stage 1 template 被解析
- **THEN** 输入 MUST 仍为 channel 0=RA、channel 1=DA 的 `[B,T,2,128,64]`，两个 branch MUST 使用不共享参数分别编码后才在 Radar 模态内部融合为 `[B,T,64]`
- **AND** 旧 `radar_cnn` 及其默认调用方 MUST 保持不变；R1 MUST NOT 改写 Radar map、缓存、归一化或引入 target/CAV/GPS/天气/场景/beam-power 特征

#### Scenario: 判断是否进入组合实验
- **WHEN** J1、C1、R1 的 validation-best 与逐 epoch 结果完整
- **THEN** C1/R1 MUST 分别与同协议同预算的旧 dedicated CSI/Radar-only baseline 比较 Top-1、Top-3、ADBA、validation loss 和 train-validation gap，J1 MUST 以同数值配置的 J0 为主对照比较五个 forced-only 指标并检查强模态退化
- **AND** 只有单因素证据支持的改动 MAY 进入另行预注册的组合实验；第一轮结果不得被表述为 test 或正式 claim

### Requirement: C2 必须只筛选 4x2、40-RE spatial expansion

C2 MUST 以 C1 一层 token Transformer 为直接对照，只把固定 pattern selection 从 `[0,1]` 扩为 `[0,1,2,3]`，frequency selection MUST 保持 `[0,15]`。C2 MUST 使用 `[5,4,2]` complex 输入、独立 selection SHA256 `2d035d64f6b9ac408532040b3ff09151a8831361d81c83b1b77e218e4344a4f4`、独立 packed cache manifest 和 fresh Stage 1。它 MUST 保持 fixed CSI-only、`mmw_id_stratified_block_v1` split seed 0、train seed 1、40 epoch、batch 64、8 workers、topology、loss 与 validation-loss checkpoint selection 不变，并保持 `test_evaluated=false`、`claim_ineligible=true`。

#### Scenario: 解析 C2 4x2 template
- **WHEN** C2 tracked template 被解析并绑定本地 cache
- **THEN** selection MUST 为 patterns `[0,1,2,3]`、frequencies `[0,15]`，每帧/窗口 complex RE MUST 为 8/40
- **AND** encoder layer、fixed modality、训练预算和 loss MUST 与 C1 相同；不得同时增加频点、改变正则化或应用 J1 supervision rebalancing

#### Scenario: C2 尝试复用 2x2 packed bundle
- **WHEN** resolver 或 sidecar 发现 cache manifest、packed metadata 或 selected tensor shape 仍绑定 2x2 selection
- **THEN** 必须在 dataset/optimizer 创建前失败并要求重建 4x2 packed cache

#### Scenario: 判断 C2 是否保留
- **WHEN** C2 完成 40 epoch 且 validation-best 已发布
- **THEN** 必须与 C1 validation-best 比较 Top-1、Top-3、Top-5、ADBA、validation loss 和逐 epoch train-validation gap
- **AND** 不得用后验 peak Top-1 替换预注册 checkpoint，也不得访问 test 或直接启动 J1+C2 组合

### Requirement: J2 必须只验证支持的 CSI 配置在联合 Stage 1 中的可用性

J2 MUST 以 J0 joint BF16 control 为联合配置基础，只把 sparse CSI 改为 C2 的 4x2/40-RE selection 与一层 token Transformer。J2 MUST 保持 `lambda_unimodal=1.0`、31-subset schedule、topology prototype supervision、其他四个 encoder、split seed 0、train seed 1、40 epoch、batch 64、8 workers、BF16 与 validation-loss checkpoint selection 不变。Stage 1 MUST fresh-start 并使用独立输出目录；用户显式授权后，系统 MAY 通过 fail-closed `continue-pipeline` 自动运行同一 J2 lineage 的 Stage 2、validation gate 与 Stage 3。全部阶段 MUST 保持 `claim_ineligible=true` 和 `test_evaluated=false`。

#### Scenario: 解析 J2 联合 Stage 1
- **WHEN** J2 tracked template 被解析并绑定本地 cache
- **THEN** sparse CSI MUST 使用 patterns `[0,1,2,3]`、frequencies `[0,15]`、`[5,4,2]` complex 输入和 `SparsePilotEncoder.num_layers=1`
- **AND** AMP、loss、temporal missing、非 CSI encoder 与训练预算 MUST 与 J0 相同；不得应用 J1 的 `lambda_unimodal=5.0`

#### Scenario: 固定配对 DataLoader 顺序
- **WHEN** J2 显式配置逐 split DataLoader generator seed
- **THEN** train/validation MUST 分别使用 J0 已记录的 `3702095051185301119` 与 `5941928843505026558`
- **AND** run metadata MUST 记录 explicit seed、experiment seed、split 和 J2 的实际 dataset fingerprint；未显式配置的其他运行 MUST 保持原有 fingerprint-derived seed 行为

#### Scenario: 判断联合训练是否保留候选能力
- **WHEN** J2 validation-loss checkpoint 已完整发布
- **THEN** evaluator MUST 在同一 validation identity 上报告全部 31 个非空 mask，并单独报告五个 forced-only 的 Top-1、Top-3、Top-5 与 ADBA
- **AND** forced-only 或任一后验 peak 指标 MUST NOT 用于替换 checkpoint或读取 test；用户对三阶段链的显式授权 MUST 独立于这些后验指标

#### Scenario: 夜间自动续跑 J2
- **WHEN** 用户显式授权 J2 在 Stage 1 后自动进入后续阶段
- **THEN** continuation MUST 等待完整 Stage 1 状态与 validation-best checkpoint，并使用 J2 专属 Stage 2/3 template 保持 4x2 selection、一层 CSI encoder、DataLoader generator seed、protocol、topology、batch 和 worker identity
- **AND** Stage 3 MUST 只在既定 Stage 2 validation gate 通过后启动；缺失专属 template、任一 identity 漂移、阶段失败或 gate 不通过 MUST 停止 pipeline且不得访问 test

### Requirement: sensing-guided probing diagnostic 必须隔离策略与 radio ground truth

系统 MUST 提供不训练模型的 validation-only beam probing feasibility diagnostic。主诊断 MUST 绑定预注册 J2 seed 1 topology-on analytic PCPF Stage 3 validation-best checkpoint，以及与该 checkpoint、`mmw_id_stratified_block_v1` seed 0 validation identity、正式 ULA-DFT topology 完全匹配的无界 prediction evidence；MUST 只评估 `image_only`、`radar_only`、`gps_only` 与 `lidar_only`，并验证其 availability 确实只开放对应 sensing 模态。结果 MUST 保持 `claim_ineligible=true`、`outer_test_accessed=false`，不得构建 test loader或根据结果切换 checkpoint。

Local-K candidate selection MUST 只依赖模型 `pred_beam`、K 与经审计 codebook topology；Uniform-K MUST 只依赖 K、offset 与 codebook size；Oracle-Local-K MUST 使用独立明确命名的函数且仅作为 claim-ineligible upper bound。完整 64-beam power、GT、CSI/channel 或未 probe gain MUST NOT 进入 Local/Uniform selection。radio-ground-truth simulator MAY 私有缓存完整 validation power vector，但其 `probe` 接口 MUST 只返回显式请求的 K 个 beam measurement，最终 beam MUST 只从这 K 个 measurement 中选择。

#### Scenario: 运行 severe-single limited probing diagnostic
- **WHEN** evaluator 对 K=3/5/7/9 运行 Direct、Local、multiple-offset Uniform、Oracle Local 与 Full-64
- **THEN** Local/Oracle MUST 使用审计确认的 ULA-DFT phase-cycle modulo-64 邻接，Uniform MUST 枚举全部 64 个 circular translation offset、保证每个 beam 恰好进入 K 个网格，并以 offset mean 为主结果
- **AND** 报告 MUST 输出逐 mask、Single Macro、Single Worst 的 Top-1、现有 normalized gain、GT coverage 与 probe budget，并保存样本级 probe indices/final beam 和 offset-level mean/std/best/worst

#### Scenario: candidate policy 尝试读取 oracle 信息
- **WHEN** Local/Uniform candidate builder 收到 GT、channel、CSI 或完整 beam-power vector，或 simulator 返回未请求 beam
- **THEN** API/断言 MUST 阻止该路径，诊断不得生成可用报告

#### Scenario: validation label 与 radio ground truth 漂移
- **WHEN** Full-64 argmax 不等于 GT，或无噪声 probe 策略的 correct 与 GT coverage 不一致
- **THEN** diagnostic MUST 失败并报告 label/power/tie 漂移，不得静默汇总
