## ADDED Requirements

### Requirement: PCPF-T 默认四模态且可显式追加历史 sparse CSI

系统 MUST 默认以 `image、radar、gps、lidar` canonical order 接受五帧历史并预测一个 64 类未来 beam。只有配置显式声明 `use_sparse_csi=true` 时，系统 MAY 在末尾追加同一五帧历史窗口的固定 2x2 sparse CSI 作为第五模态。模型、loss 和风险 target MUST NOT 读取当前/未来 CSI、未来 channel、path、beam power、历史 beam index、天气、场景、domain、corruption type 或 severity；风险估计前 MUST NOT 执行跨模态 attention 或 feature concat。全部当前开发结果 MUST 标记 `claim_ineligible=true`，MMW test MUST 默认封存。

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

四项 normalization mean/std 与 Stage 3 的 `mean_train_risk_m` MUST 只遍历 train dataset 拟合并作为 buffer 冻结；validation/test MUST 不更新它们。`R_star_m` MUST 使用 detached deterministic unimodal probability 与当前 topology 的 normalized circular distance，Dmax MUST 来自 topology，且 unavailable 模态 MUST 排除。

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

系统 MUST 复用 Full Top-1/3/5、Single Macro/Worst、全部非 Full mask Macro/Worst、Within-3、circular MAE、每模态 Missing/Single、sunny/rainy/foggy 和 15-domain macro/worst。默认四模态 MUST 评估 15 个非空 mask，sparse CSI opt-in MUST 评估五模态全部 31 个非空 mask。每个 mask/weather MUST 额外输出 weight mean/std/percentile、相对 static prior 偏差、missing weight max、effective modality count、risk-weight Spearman、真实风险排序一致率、NLL/Brier/ECE/reliability data 与 temperature。confident-but-wrong MUST 使用每模态 train-only 90% confidence threshold。

#### Scenario: 运行 validation mask evaluator
- **WHEN** evaluator 使用一个 validation-best checkpoint 和同一 validation split
- **THEN** 默认所有 15 个或 opt-in 所有 31 个非空 mask MUST 使用同一模型参数与单模态 evidence
- **AND** 输出 MUST 包含 claim eligibility、checkpoint、split、weather/domain 和 normalization provenance

### Requirement: 配置、数值和 smoke 必须失败关闭

PCPF parser MUST 拒绝未知字段、负 loss/risk 系数、非正 temperature、`d_model % num_heads != 0`、非五帧 seq、非法 stage、缺少后续 stage checkpoint/gate 的训练请求。trajectory sparse-CSI Stage 1 MAY 显式 fresh start；若提供初始化 checkpoint，则其训练协议与 normalization MUST 与当前 protocol fingerprint/validation 隔离契约一致。risk/softmax/exp/log/KL MUST 在 FP32 执行，即使主模型为 BF16。实现 MUST 提供 static/focused tests、synthetic forward/backward、真实 MMW 单 batch Stage 1、Stage 1 假 checkpoint的 Stage 2 和 Stage 3 smoke，并报告 shape、loss 分量、关键梯度、missing weight、row-sum、NaN/Inf 与 GPU peak memory。

#### Scenario: BF16 风险 forward
- **WHEN**主 expert tensor 为 BF16
- **THEN** risk component、raw risk、fusion score 与 KL 的内部输出 MUST 为 FP32
- **AND** canonical logits MAY 在稳定计算完成后 cast 回模型 dtype

#### Scenario: 非法配置字段
- **WHEN** PCPF loss/model config 包含未声明字段
- **THEN** parser MUST 在模型训练前列出未知字段并失败

### Requirement: 历史 sparse CSI 必须固定、复数且可审计

启用 sparse CSI 时，系统 MUST 对每个历史 frame 使用固定 pattern index `[0,1]` 与 frequency index `[0,15]`，得到 `[5,2,2]` complex tensor；mother grid MUST 为 `[5,32,16]`，每帧抽样率 MUST 为 `4/(32*16)=0.78125%`。正式路线 MUST 绑定当前 `mmw_id_stratified_block_v1` seed manifest 的实际 train/validation windows，selection descriptor、selection SHA256、probe codebook logical/file SHA256、physical frequency offset、历史 frame id、channel path identity、protocol/version、block size、manifest/source/window hash 与 cache identity MUST 写入 resolved config 或 sidecar metadata。训练前 cache scan MUST 只遍历 train/validation 并记录 `test_evaluated=false`，并 MUST 发布由完整 block protocol identity、自身 SHA256 和原内容寻址 cache key 绑定的 packed `[N,2,2]` complex cache；旧 split cache MUST 失败且要求重建。正式 dataset MUST 严格命中该 bundle，不得在 worker 中回退到 source channel 计算。正式 resolver 还 MUST 绑定并严格校验 RGB/LiDAR 帧缓存和同一 block protocol 的 GPS coordinate cache。生成路径 MUST 不加入 AWGN、pilot dropout、随机 corruption 或任何 current/future CSI。真实 SNR 不可得时 MUST 记录 `snr_available=false`，不得随机生成或用常数冒充。

#### Scenario: 编码历史 sparse CSI
- **WHEN** batch 提供 `[B,5,2,2]` complex pilot 与 `[B,5,2]` pattern id/frequency position
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

正式 trajectory R0--R7 MUST 绑定已通过完整 domain、64-beam label、power replay 与有限数值检查的 `ula_dft_phase_cycle_v1` audit。resolved config、模型构造、Stage 2/3 initialization、validation-best checkpoint、gate 与 matrix MUST 绑定相同 descriptor SHA256、audit 文件 SHA256、protocol audit、split seed、train/validation identity 和 experiment seed；任一文件缺失、内容漂移或 identity 不匹配 MUST 在训练或报告前失败。`cyclic_index_v1` 运行 MUST 保持 `claim_ineligible=true`，且不得通过修改 config 或报告 metadata 后验升级。复用 observation evidence 时 MUST 额外匹配主 checkpoint SHA256、control checkpoint SHA256、topology、protocol 与 seed；所有 mask MUST 与 Full mask 使用相同样本和顺序。

#### Scenario: 旧 cyclic checkpoint 搭配 formal config
- **WHEN** initialization 或 evaluator 尝试把 `cyclic_index_v1` checkpoint 加载到声明 `ula_dft_phase_cycle_v1` 的模型
- **THEN** 系统 MUST 在加载 state dict 或写出 formal provenance 前拒绝

#### Scenario: 复用不同 checkpoint 的 observation cache
- **WHEN** `--reuse-evidence` 指向由不同主 checkpoint、control checkpoint、protocol、topology 或 seed 生成的 cache
- **THEN** evaluator MUST 拒绝复用且不得把 cache 重标为当前 checkpoint 证据

#### Scenario: mask 样本配对漂移
- **WHEN** 任一 validation mask 缺少、重排或混入不同于 Full mask 的样本 identity
- **THEN** gate 或 matrix MUST 在聚合指标前拒绝报告

### Requirement: R0--R7 与 D0--D3 必须使用相同专家和配对样本

系统 MUST 报告在同一 `mmw_id_stratified_block_v1` seed manifest 上重新训练的 R0 四模态 PCPF-T 参考、R1 五模态联合训练 checkpoint 强制 CSI 缺失、R2 五模态 uniform、R3 五模态 train-only static prior、R4 五模态 direct Router control、R5 五模态 `cuaf_local_adaptation`、R6 五模态当前 analytic PCPF 和 R7 同一联合 checkpoint 的 CSI-only。R0--R7 MUST 使用同一 validation identity、split seed、train seed、manifest/window hash 与样本顺序；R1--R7 还 MUST 共享 mask、Stage 1 expert checkpoint、Stage 2 probability/risk checkpoint、unimodal logits 和 temperature calibration 基础。旧 split 或 clean-inner 结果 MAY 作为带明确 legacy 标签的背景展示，但 MUST NOT 作为 paired R0 或初始化 checkpoint。每个 run MUST 保存样本级 identity/group、label/prediction、每专家 logit/probability/真实 circular error/risk/risk component/static weight/dynamic weight、availability 与 CSI quality字段。

#### Scenario: 运行机制诊断
- **WHEN** evaluator 完成五模态 31-mask validation
- **THEN** D0 MUST 使用原始逐样本 dynamic risk，D1 MUST 在相同 domain 与 mask 内确定性打乱 sample risk，D2 MUST 用相同 domain 与 mask 的平均 risk 替换 sample risk，D3 MUST 使用 static prior
- **AND** 四者 MUST 在相同 cached unimodal probability、availability 与样本顺序上重新融合并报告 paired 指标；risk/error、component、confident-wrong、weight transfer 与分布统计 MUST 作为独立机制诊断输出，不得冒充 D0--D3

#### Scenario: 计算不确定性区间
- **WHEN** evaluator 汇总主结果与 paired 差值
- **THEN** bootstrap MUST 以审计通过的独立 group key 重采样，并记录 seed、次数和 group 数
- **AND** 不得把同一 trajectory/window 的帧视为独立样本
