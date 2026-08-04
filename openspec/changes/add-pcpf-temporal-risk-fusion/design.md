## Context

当前 U0 的四个 encoder 均以五帧序列为输入并返回 `[B,T,D]`，随后由 `UMaskBeamJEPA` 做 masked mean、构造单模态 prototype logits，再用 supervised Router 融合。训练缺失契约的真实形状是 `modality_temporal_mask=[B,T,4]`，`available_modalities=mask.any(dim=1)`；`BeamPrototypeBank` 持有唯一可学习 `[64,64]` prototype，使用归一化 feature/prototype 的 cosine logits，现有 `prototype_alignment_loss` 已覆盖 fused/modality feature、availability mask 和 topology soft target。

通用 trainer 在模型构造和 initialization checkpoint 加载后创建 optimizer，通过 training extension 替换 base loss，并按绑定协议的 validation loss 保存 `best.pth`。15-mask evaluator 已覆盖 Full、drop-1、drop-2、Single、Within-3、circular MAE、ECE 和宏/最差聚合；MMW pooled dataset metadata 提供 weather/domain。PCPF-T 默认四模态路径不读取 CSI；opt-in 路径只复用固定 channel simulator/codebook 原语，并直接从 trajectory protocol 中当前样本的五帧历史 channel 引用生成 sparse CSI。

历史工作树曾同时存在多个 active change。此次收敛将它们按真实完成/停止状态保存到仓库外快照，当前工作树只保留 PCPF-T change；非 PCPF 本地实验 source 由 Git 历史追溯，不继续在线维护。

## Goals / Non-Goals

**Goals:**

- 在 canonical `image, radar, gps, lidar` 和五帧 mask contract 上实现独立注册的 PCPF-T。
- 让每个模态先独立经过共享 Temporal Transformer，再查询同一个 `BeamPrototypeBank`。
- 以低自由度、非负系数公式学习 `U_var/U_proto/U_temp/U_conflict` 到 topology risk 的映射，并用固定 precision-style 公式融合概率。
- 复用通用 optimizer、training extension、checkpoint、validation-best、MMW protocol 和 15-mask metrics。
- 对训练阶段、train-only 拟合状态、checkpoint 来源角色、Stage 2 gate、FP32 风险计算和 missing 权重做 fail-closed 验证。
- 提供无需 test 或长训练即可执行的 synthetic/one-batch smoke 和 resolved-config workflow。
- 在显式 opt-in 时把固定历史 sparse CSI 作为第五个 temporal expert，并覆盖全部 31 个非空 availability subset。
- 用同一验证集完成公平 R0--R7、D0--D3 和分组 bootstrap，全部结果保持 `claim_ineligible=true`。

**Non-Goals:**

- 不改变、替换或删除 U0 Router，也不把 PCPF-T 参数加入 U0。
- 不把 PCPF-T 变成第四个 canonical MMW recipe或新增 public CLI。
- 不读取当前或未来 CSI/channel、path、beam power、历史 beam、天气、场景或 corruption metadata 作为模型输入/target；唯一例外是 opt-in 的样本自身五帧历史 sparse CSI。
- 不实现 frame-level supervised CE；数据没有独立逐帧未来标签时，不用同一个未来标签伪造五个目标。
- 不默认加入模态间直接对比、跨模态 attention、feature concat 或任意四维动态 Router。
- 本 change 的 MMW 协议定义 70/15/15，但默认不读取 test，也不运行 multi-split，且不据 validation/test 结果事后修改 Stage 2 gate；开发阶段只使用 `split_seed=0`，训练随机性由独立 `train_seed` 控制，正式长训练不属于本轮任务。

## Decisions

### 1. 独立整模型而非继承 UMaskBeamJEPA

新增 `pcpf_temporal_risk_fusion`，复用 registry encoder 配置和 `BeamPrototypeBank`，但不继承 `UMaskBeamJEPA`。继承会无条件创建 reliability head、classifier 和 supervised Router，既污染 checkpoint，也使“解析式动态融合无自由 Router”无法审计。公共 import 只注册 class；只有 recipe 显式声明该 type 时才实例化参数。

替代方案是继续给 U0 增加 opt-in 分支；这会扩大 U0 forward 和 state dict，拒绝采用。

### 2. 一个共享 Temporal Transformer，逐模态独立进入

`SharedTemporalTransformer` 接受 `[B,T,M,D]` 与 `[B,T,M]`，其中默认 `M=4`，历史 sparse CSI opt-in 时 `M=5`。它为每个模态应用独立 input LayerNorm/可选线性 adapter，再将张量重排为 `[B*M,T,D]`。共享 time embedding `[T,D]`、modality embedding `[M,D]`、T-CLS 和两层 `TransformerEncoder`；T-CLS 永远可见，frame padding mask 来自真实 temporal mask，`causal=false`。

输出 frame token `[B,T,M,D]`、CLS `[B,M,D]`、valid fraction 和 availability。整模态缺失时允许 Transformer 完成计算，但随后将该模态的 frame/CLS 显式置零，防止 T-CLS 或 embedding 产生伪证据。风险估计前没有跨模态 attention 或 feature concat；唯一跨模态量是基于 detached/显式 probability 的弱 conflict 标量。

### 3. 单一 prototype bank 同时承担专家预测和语义对齐

模型只创建一个 `BeamPrototypeBank`。Stage 1 的 deterministic unimodal logits 来自 temporal CLS；`prototype_alignment_loss` 直接复用 fused/modality soft topology alignment，不复制 InfoNCE。新增的 unimodal loss 对每个样本的可用模态先算 hard CE 与 soft topology CE，再按该样本可用模态数归一化。

Stage 1 默认 uniform 概率融合；static learnable prior 仅作为显式 control。概率融合结果以 `log(p_fused)` 暴露为 canonical logits，避免把概率加权误写成 logits 加权。

### 4. 共享概率嵌入和显式四项风险

Stage 2 使用一个共享 `ProbabilityEmbeddingHead`：零初始化 `DeltaMu` 末层使 `mu=h`，零权重且 bias=-4 的 logvar head 产生 `[B,M,D]`，并 clamp 到 `[-8,4]`。训练可用重参数样本仅服务 preserve loss；deterministic `mu` logits 始终用于 `R_star`、风险与评估。

四项风险全部在 autocast disabled 的 FP32 block 中计算：

- `U_var = mean(exp(logvar))`；
- `U_proto = 1 - max cosine(mu, prototype)`；
- `U_temp` 从有效 frame prototype distribution 的 circular mean 得到角度序列，至少三帧时先用最短 circular difference 展开，再拟合一阶趋势并以 `pi^2` 归一化残差；不足三帧为零且 `temp_valid=false`；
- `U_conflict` 为当前模态与其他可用模态均值分布的 JS divergence；Single 时为零。

`raw_risk=softplus(sum softplus(rho_x)*normalized(U_x)+bias)`，四个系数默认跨模态共享。禁用消融分量时在公式入口置零且不改变其余分量定义。

### 5. train-only 状态通过 opt-in stage preparation 拟合

通用 trainer 在 initialization checkpoint 加载后、optimizer 创建前，仅当模型实现 `prepare_training_stage(...)` 时调用该方法。PCPF-T 用 train dataset 的顺序只读 loader：Stage 2 拟合并冻结四项风险的 mean/std；Stage 3 以 deterministic detached unimodal probability 和真实未来 label 计算每模态 `mean_train_risk`。该 pass 不读取 validation，保存 JSON 与 model buffers，并恢复原 loader RNG 状态。

Stage 2 初始 logvar 恒为 -4，因此 `U_var` 在 preparation 时退化为常数。`risk.normalization_epsilon` 是风险分量的标准差下限而非机器精度 epsilon，正式配置固定为 `0.01`；拟合标准差低于该值时保存并使用下限，避免随后训练 probability head 时把微小 `U_var` 变化放大到 Softplus 饱和区。

smoke 可显式限制 preparation batches；正式配置必须为全 train pass。U0 和保留 baseline 没有该方法，所以不会多走 forward、修改 RNG 或改变 optimizer。

替代方案是在每个 Stage 2 batch 在线更新统计；这会让 normalization 与风险头共同漂移且 validation 时点难以复现，拒绝采用。

### 6. 固定的三阶段冻结和 fail-closed checkpoint 来源

模型构造时按 stage 设置 `requires_grad`，因此 optimizer 只看见允许参数；`train()` 会让全部冻结 module 保持 eval。启动 metadata 输出完整 trainable parameter names/count，extension 再断言实际集合等于预期集合：

| Stage | 训练参数 | 冻结参数 |
| --- | --- | --- |
| `stage1_expert` | encoders、encoder projections、temporal input adapter/共享 Transformer、prototype bank；sparse CSI opt-in 时额外训练 CSI encoder/projection/第五星 embedding；static control 时仅额外 global prior | probability/risk、temperature/tau/eta、direct Router |
| `stage2_risk` | probability embedding、risk coefficients/bias | encoders、projection、Temporal Transformer、prototype、static prior、temperature/tau/eta、Router |
| `stage3_fusion` | 每模态 temperature、tau、可选 eta；显式配置时才微调共享 risk coefficients | expert/prototype/probability head，其余 control |
checkpoint payload/sidecar 增加通用 `model_metadata`，PCPF-T 记录 stage、fusion mode、train-only stats、claim eligibility 和 gate 状态。初始化请求可声明 `expected_source_training_stage`；Stage 2/3 分别只接受 Stage 1/2 的 `validation_best`，不匹配时在加载 optimizer 前失败。通用 `best.pth` 保持不变，并为 PCPF-T 创建 `stage1_best.pth` 等只读相对 symlink。

### 7. Stage 2 gate 是独立只读评估产物

Stage 2 loss 使用 detached `R_star=sum p_mu(k)*normalized_topology_distance(k,y)`、masked Huber、margin pair ranking、很小 Gaussian KL 和 sampled/deterministic preserve topology CE；不使用 fused CE 更新风险。

评估器从同一 checkpoint 输出 overall、modality、weather、domain、mask group 的 Pearson/Spearman、decile/calibration 和 confident-but-wrong 诊断。门槛来自 tracked config，报告只计算 pass/fail 和失败原因，绝不写回阈值。Stage 3 resolved config 必须绑定 gate JSON 与 SHA256 且 `stage2_gate_passed=true`；普通 stage launcher 不自动串联启动 Stage 3，显式的本地 `continue-pipeline` 仅可在无界 gate 通过后解析并启动 Stage 3。

### 8. Stage 3 只做概率校准和解析融合

模型保存 train-only `mean_train_risk_m`。`eta` 为非负标量，`a_m=exp(-eta*mean_train_risk_m)`；`T_m=Tmin+softplus(t_m)`，`tau=tau_min+softplus(raw_tau)`。在 FP32 中用 log-score `log(a_m)-risk/tau` 加 availability masked softmax，避免 `exp` 下溢，并保证 missing weight 为零、Single weight 为一、行和为一。

`p_fused=sum w_m*softmax(logits_m/T_m)`。Stage 3 loss 为 NLL 加可选 topology soft CE，不将 validation 结果输入 forward。

### 9. 对照共享 expert 初始化，但不污染主模型

Uniform、Static Prior、PCPF-T 和 `no-*` 直接通过解析 fusion mode/分量开关替换。Direct Router 是仅在 A2 control config 中创建的旧式标量特征 MLP，metadata 明确 `control_only=true`；A4 主模型没有该 module。CUAF 无法由任务简称唯一定位逐项公式，因此实现只命名为 `cuaf_local_adaptation`，使用 normalized entropy、margin risk 与 JS conflict 的固定解析组合，不宣称论文复现。

动态替换评估缓存同一次 A4 forward 的 unimodal logits 与风险分量，再应用 A0--A3 各自 validation-best checkpoint 中已训练的 temperature、tau 或 Router 参数；不得为 control 重跑 encoder。所有 control checkpoint 必须与 A4 具有相同 Stage 1 expert fingerprint，且路径、SHA256、fusion mode 和 checkpoint role 写入 matrix provenance，否则拒绝比较。

### 10. 本地配置和评估面，不扩大 canonical surface

tracked template 位于 `tools/configs/pcpf/`，通过现有 `_base_`、config loader、MMW protocol 注入和严格 PCPF parser 解析；不在 `configs/mmw/` 增加 canonical recipe。一个 `tools/run_pcpf.py` 负责 trajectory protocol prepare、resolve、preflight、train 和 smoke，一个 `tools/eval_pcpf.py` 负责 gate、默认 15-mask/opt-in 31-mask、替换和诊断；不复制 trainer。默认四模态产物保留在 `outputs/pcpf_temporal_risk/`，本轮 sparse-CSI resolved config、stats、reports、checkpoint 和 smoke 独立写入 `outputs/pcpf_sparse_csi_router_v1/`。

### 11. 历史 sparse CSI 是固定可审计的 dataset sidecar

PCPF sparse-CSI 配置只允许 `use_sparse_csi=true`、五帧历史和 TSPC-V2 固定 2x2 选择：pattern `[0,1]`、frequency `[0,15]`，每帧 4 个 complex RE。正式路线必须直接加载当前 `mmw_id_stratified_block_v1` seed manifest 的 train/validation domain CSV，并从当前样本行的 `csi1..csi5` 引用确定性生成；验证 channel 文件 stem 与历史 frame id 一致且最后历史帧早于 target。不得跨协议、split role 或 block join。sidecar 记录 protocol/version、block size、manifest/source/window hash、split seed、mother shape、selection descriptor/SHA256、codebook logical/file SHA256、frequency offset、history id、cache identity 和 `snr_available=false`。训练前只扫描 manifest 声明的 train/validation 样本补齐同一内容寻址 cache，旧 split-specific cache 必须失效；test 只允许独立显式最终评估读取且绝不进入 cache scan。

Sparse CSI encoder 保留 real/imag/log-magnitude 和有效位置；真实 SNR 缺失时不接受随机或常数伪造。SNR 只能作为可选质量诊断，不参与 CSI feature、logit、risk target 或 fusion。生成路径不得调用 AWGN、pilot dropout、synthetic corruption 或 future/current channel。

### 12. sparse-CSI Stage 1 必须 fresh start

`use_sparse_csi=false` 时模型不得创建 CSI 参数。trajectory 五模态 Stage 1 必须 fresh start，不提供四到五模态 checkpoint 迁移；Stage 2/3 只接受同一五模态 trajectory protocol 下前一阶段的 validation-best checkpoint。

### 13. 五模态训练与评估覆盖全部非空子集

opt-in 训练 schedule 以 global sample position 和 epoch seed 确定性轮转 31 个非空 bitmask；一个 epoch 内各 subset 计数差不超过 1，并写入训练日志。mask 同时作用于 sensing temporal tensor 和 CSI frame mask；缺失 CSI tensor 显式清零，缺失 probability/risk/weight 为零，Single weight 为一。R0 必须是在同一 trajectory protocol/split seed 上重新训练的四模态参考；旧 split 结果只作带明确 legacy 标签的背景，不进入 paired R0--R7。R1 是五模态联合 checkpoint 强制 CSI 缺失；R2--R6 分别是 uniform、train-only static prior、direct Router control、`cuaf_local_adaptation` 和当前 analytic PCPF；R7 是同一联合 checkpoint 的 CSI-only。R0--R7 共享 trajectory validation identity、split seed 与 train seed，R1--R7 额外共享 Stage 1/2 checkpoint、unimodal evidence、样本顺序、mask 和 temperature calibration 基础。

### 14. 机制诊断只使用固定验证证据

评估器必须输出可复算的样本表，至少包含 sample/group identity、label/final prediction、每专家 logits/probability/prediction/真实 circular error/risk/risk components/static weight/dynamic weight、availability、confidence/correctness、CSI norm/valid fraction/quality/SNR availability。D0 使用原始逐样本 risk；D1 在相同 domain+mask 内以固定 seed 打乱 risk；D2 用相同 domain+mask 的平均 risk；D3 使用 static prior，四者只在相同缓存 evidence 上重算融合。另行报告逐模态/天气/domain/CSI presence/cardinality 的 risk-error 与 component-error 相关性、AUROC、confident-wrong、weight 分布和 Full-to-mask transfer。置信区间必须按可用独立 group 做 paired bootstrap；无 trajectory id 时使用审计通过的稳定 sample group key，不得把 mask 重复或帧伪装为独立样本。

### 15. 正式 trajectory 运行使用严格派生缓存与固定物理 batch

trajectory sparse-CSI resolver 直接绑定现有 `outputs/cache/MMW` RGB/LiDAR 严格帧缓存和当前协议的 train/validation GPS coordinate cache；任一缓存目录、metadata、协议 fingerprint 或 coverage manifest 不匹配时，在长训练前失败。CSI cache scan 除逐 channel 内容寻址 cache 外，再发布一个只含当前 seed 0 train/validation 唯一历史 channel 的紧凑 packed bundle；其精确数量由当前 manifest 重建并写入 provenance，不在代码中硬编码。bundle 保存确定性 `[2,2]` complex selection、绝对 channel path、原内容寻址 cache key、selection/codebook/cache spec、protocol fingerprint、manifest version、split seed、split manifest 和自身 SHA256；dataset 在父进程中一次校验并加载，worker 只做内存查找，不再重复打开 source channel 或逐文件 cache。

正式 sparse-CSI 模板使用 batch 64、8 workers；Stage 1/2/3 与 R1--R7 沿用同一物理 batch 约定，CLI 仅在显式给出时覆盖。batch 64 先经过真实 CUDA 单步显存 smoke，失败时必须显式记录回退配置并重新 fresh start，不能从不同 batch 的 checkpoint 续训。

### 16. 正式证据必须绑定经审计的 ULA-DFT topology descriptor

当前 `cyclic_index_v1` trajectory 配置只服务数值、流程和运行稳定性开发；由它产生的 checkpoint、gate、matrix 和诊断 MUST 保持 `claim_ineligible=true`，不得进入正式 R0--R7、论文 claim 或归档完成证据，也不得在事后补 metadata 将既有运行升级为可宣称证据。

正式 R0--R7 启动与本 change 归档前，resolver MUST 绑定一份经审计的 ULA-DFT topology descriptor。descriptor MUST 具有稳定 ID 与内容 SHA256，并明确 64 类 beam 的物理顺序、circular distance 定义及其 codebook/protocol 依据；resolved config、各阶段 checkpoint、Stage 2 gate 和 R0--R7 matrix MUST 记录相同 descriptor identity/SHA256，任一缺失或不匹配 MUST 在训练或比较前失败。源码测试 MAY 使用等价的 64 类 cycle 验证算法，但该测试 topology 不构成正式运行 provenance。

audit 文件内容与 SHA256 在 resolve、preflight、train、模型构造和 evaluator 加载时重新验证；不能仅凭 topology ID 或 64 位字符串形状产生 formal eligibility。通用 initialization 与 evaluator 都比较 source checkpoint 和目标模型的 topology identity，并同时核对 checkpoint 记录的 protocol/seed。observation cache 记录主/control checkpoint SHA、protocol、topology 与 seed，复用时做精确匹配；报告聚合前按每个 mask 校验与 Full mask 的 sample identity/order 配对，阻断报告失败后的跨 checkpoint 重标。

### 17. trajectory 数据画像必须独立于模型选择且默认封存 test

新增本地 `tools/analyze_mmw_trajectory_dataset.py`，直接读取 canonical seed 0 manifest 与对应 audit 元数据，开发画像只打开并重新校验 manifest 绑定的 train/validation CSV。它必须确认 test CSV 存在于 manifest 但未加载，并记录 `test_evaluated=false`。

分析入口在不提供 test loader 的前提下重算 manifest identity，并校验正式 protocol/version/split seed/block size、manifest/source/window hash、role、实际 group/window count、split hash、audit identity 与 train/validation block/base/weather/window-frame 零交集项。正式信号扫描默认 strict；sparse-CSI 还必须由调用方提供 resolved config 记录的 packed-cache SHA256，并重新校验固定 selection/schema/protocol/path set。输出只允许位于专用 ignored `outputs/mmw_trajectory_dataset_analysis/` 根，写入和 artifact inventory 拒绝 symlink，inventory 只覆盖本次启用模态与实际生成文件；任一声明产物缺失时保持 `running` manifest 并失败，只有报告、图表和逐件 SHA256 inventory 全部完成后才发布 `passed`。

分析按唯一资源去重后覆盖 split/group/domain/weather/agent 组成、64 类分布与 shift、滑窗重叠和 label 自相关有效样本量、历史 beam-power persistence/趋势等捷径基线、五帧 geometry/GPS 动态、future beam-power 标签一致性与模糊度、RGB/Radar/LiDAR/sparse-CSI 的有限性、质量和 train-validation shift。分析还必须按同 scenario、agent 与时间序号配对 train/validation，量化跨天气 target、geometry 和各历史模态确定性签名的重合；该配对只诊断 group ID 隔离是否仍保留可记忆的轨迹内容，不得作为模型输入或合法 baseline。future beam power、geometry、天气、场景、序号和诊断标签只可标记为 `diagnostic_only`，不得因此进入模型输入或风险 target。

轻量 diagnostic probe 使用 train-only 拟合、固定 seed/预算；validation 只做一次只读外部刻画，不参与 early stopping、阈值或特征选择，test 不读取。probe 只回答低分辨率确定性签名中是否存在标签信息或 split 可分性，不得冒充 baseline、模型改进或论文 claim。报告必须明确三个 split 都含相同 trajectory 的不同时间 block，因此不得把 validation 高分解释成未知 trajectory、场景或天气泛化。

Geometry、weather、scenario 与 route-position pairing 只参与不可拟合的描述和签名重合诊断，绝不进入 probe feature。连续 shift、feature-space OOD、配对重合和 probe 排名以 trajectory group 等权统计为主，并列保留 window-micro 描述；配对签名 RMSE 的训练尺度先在每个 trajectory group 内汇总再按 group 等权估计，不得由大 group 的重叠 window 主导。异步资源错误和 probe 行稳定排序，缺失资源不得被解释为 exact signature 或有效 beam baseline。

全部中间签名、CSV、JSON、Markdown 和图表写入 ignored `outputs/mmw_trajectory_dataset_analysis/`，并记录协议/manifest/window identity、train/validation 文件 SHA256/count、参数、seed、代码版本、`claim_ineligible=true` 与 `test_evaluated=false`。每条建议必须引用报告中的量化证据并区分数据事实、diagnostic probe 观察和待验证假设。

### 18. 三阶段续跑复用现有命令并失败关闭

`tools/run_pcpf.py continue-pipeline` 只编排现有 resolver、共享 trainer 与 `tools/eval_pcpf.py gate`，不复制训练或评估实现。它从 Stage 1 resolved config 推导同系列 Stage 2/3 template、协议、audit、物理 batch、worker、seed、输出根与运行名，并等待该 Stage 1 的 `run_status.json` 进入 `complete`。进入下一阶段前必须验证前一阶段 `last.pth` 已达到配置 epoch、stage-specific validation-best checkpoint 已完整发布且 metadata stage 匹配。

Stage 2、gate 与 Stage 3 分别在独立子进程中运行，使 CUDA 与 dataloader worker 在阶段边界释放。gate 必须是同一 Stage 2 validation-best 上的无界只读评估；返回失败、报告不通过、checkpoint/protocol/seed 漂移、运行状态失败或 stale PID 时续跑器立即非零退出，不得解析或启动 Stage 3。该动作只用于显式启动的本地长训练，不修改 public CLI、系统启动项或 outer-test 封存状态。

续跑器在复用已有 run 前比较 run-local resolved config 与 checkpoint resume config 的闭集 lineage，覆盖 protocol/audit、seed、topology、物理 batch/worker、训练预算、initialization checkpoint 与 gate SHA；同名旧 run 不得仅凭 stage/epoch 被复用。已有 gate 也必须匹配当前 Stage 2 checkpoint 与完整 lineage，续跑入口使用文件锁阻止同一配置的并发编排。

### 19. MMW 使用 verified weather-bound block assignment

协议构建读取 15 个 strict sequence index，并使用每行 `seq_index`、`history_frame_ids_json` 与 `future_frame_ids_json` 建立天气内物理 frame 到 `(scene_id,cav_id,base_frame_index)` 的显式映射。重叠窗口对同一物理 frame 给出的 base index 必须一致；三天气的 base index 集、对应标签和窗口跨度必须一致。frame manifest 仅用于补充并验证基础帧 inventory/source hash；不能按物理 frame 数值或 CSV 遍历顺序跨天气配对。

每条 trajectory 按 base index 每 32 点切 block，尾块保留；block assignment 先保证每 trajectory 三 role 覆盖与整数 quota，再使用 seed 控制的确定性多起点局部 swap 优化预计窗口 ratio、全局 beam TV、按 scene/domain 与 trajectory 的 train--validation/train--test TV、trajectory/scene ratio和对应 train 条件覆盖。32 点 block 在保持连续时间隔离的同时为每条 trajectory 提供多个分散 held-out block；最终窗口通过严格候选 index 在 assignment 后 materialize，只有全部历史与目标 base index 属于同一 block 才写入 role CSV。

manifest 记录 protocol version 1、manifest schema version 2、assignment algorithm、70/15/15、block/trajectory/base identity、source/window hash、完整 block 清单、全局/scene/trajectory assignment objective 与实际窗口统计。JSON/Markdown report 同时给出优化 assignment 与简单前段 block baseline 的全局及条件 TV/JSD/correlation/coverage。已有 manifest 任何 identity 不匹配时拒绝复用；只有显式 regenerate 才可重写。所有 split-specific cache 继续由 manifest hash 失效，原始内容寻址 CSI cache不删除。

### 20. 单模态能力诊断使用独立 Stage 1，而非联合 checkpoint 的强制 mask

联合 31-subset Stage 1 的 Single-mask validation 同时受到共享 prototype、其他专家训练和 mask schedule 影响，不能回答单个传感器在当前 split 上能否独立学到 beam label。诊断因此为 image、radar、GPS、LiDAR 与 sparse CSI 分别启动 fresh-start `stage1_expert`；五条运行绑定同一个 `mmw_id_stratified_block_v1` seed 0 manifest、train seed、40-epoch 预算、batch/worker、topology 与 validation-loss checkpoint selection。

`fixed_single_modality` mask 在训练时只保留指定模态的真实历史 cell，并与 source availability 取交集；若任一样本没有该模态的有效历史 cell，必须失败，不能恢复其他模态。逐 epoch validation 使用同一个固定 modality mask，31-mask 末尾评估关闭。诊断不启动 Stage 2、Stage 3、gate 或任何融合对照，不读取 test，结果保持 `claim_ineligible=true`。联合 checkpoint 的 forced-only 数字和旧 clean-inner 结果只作为背景，不与该诊断混成同一协议结论。

### 21. 拓扑原型监督与动态融合使用嵌套 2x2 消融

创新点一严格定义为 Stage 1 的 topology-aware prototype supervision，而不是 K-means 或每类多 prototype 聚类。关闭分支仍实例化同一个 64-beam `BeamPrototypeBank`，所有模态继续通过该共享 bank 产生 logits，并保留 fused/unimodal hard CE；仅将 `unimodal_soft_weight`、`lambda_proto`、`lambda_modality_proto` 置零并关闭 `use_beam_prototype_alignment`。这样反事实只移除邻近 beam topology soft target 与 fused/modality prototype alignment，不改变 head 容量、参数量或推理路径。

每个 train seed 分别 fresh-start 训练 topology supervision 开/关两条 Stage 1 专家链，并各自完成 Stage 2。每条专家链从同一个 Stage 2 validation-best 和 gate 分叉 Static Prior 与 analytic PCPF Stage 3，形成 E0=`topology off + static`、E1=`topology on + static`、E2=`topology off + dynamic`、E3=`topology on + dynamic`。创新点一主效应报告 E1-E0，创新点二在创新点一基础上的条件效应报告 E3-E1，同时报告 E2-E0 与交互项 `(E3-E1)-(E2-E0)`；不得用 `no-proto` 风险分量消融替代 topology-loss 消融。

补充实验固定 `mmw_id_stratified_block_v1` split seed 0、train seed 1/2/3、五模态 31-subset schedule、40/20/10 epoch、batch 64、8 workers、`ula_dft_phase_cycle_v1` topology 和 validation-loss checkpoint selection。四个 cell 对同一 seed 使用同一 validation identity、样本顺序与指标实现；每条专家链内 Static/Dynamic 必须共享 Stage 1/2 fingerprint。跨 topology 开/关分支的 expert fingerprint 本来就应不同，评估不得伪称同专家对照，只允许在相同 sample/group identity 上计算 seed 内 paired 差值，再跨三个预注册 seed 汇总。开发流程继续封存 test，全部结果保持 `claim_ineligible=true`。

## Risks / Trade-offs

- [Stage 2 risk 可能与 `R_star` 无相关性或常数化] -> 预注册 gate 失败即停止，不自动启动 Stage 3，也不修改阈值。
- [Stage 2 初始 `U_var` 方差为零] -> 使用预注册的 `0.01` normalization std floor，并以 preparation 后单优化步的有限梯度和非零风险回归测试防止塌缩。
- [Transformer 对全缺失模态仍能由 CLS 产生非零值] -> forward 后按 availability 显式清零并测试 probability/risk/weight。
- [FP16/BF16 下 exp/log/JS/KL 下溢] -> 风险、校准和融合统一在 FP32 中执行，输出结束再 cast。
- [train-only preparation pass 增加启动耗时] -> 正式运行完整遍历以保证统计契约；仅 smoke 允许限 batch，并在 metadata 标记不具 claim 资格。
- [Stage 1 随机、未训练 probability/risk 参数进入 checkpoint] -> 它们存在于 PCPF-T state 以支持严格 stage 初始化，但被冻结且不参与 forward/loss；checkpoint metadata 明确 inactive modules。U0 state dict 不受影响。
- [test 被开发流程意外消费] -> 默认不构建 test；仅显式 `--evaluate-test` 的最终只读路径可加载，并记录 `test_evaluated=true`，PCPF gate/选择/统计始终拒绝 test。
- [跨天气 frame id 不同导致错误绑定] -> 只接受 strict `seq_index` 与显式窗口 frame 列表导出的映射，三天气 inventory/label/span 任一不一致即失败。
- [旧 split checkpoint/cache 与新 manifest 身份不一致] -> trajectory Stage 1 只允许 fresh start，cache 必须按当前 fingerprint 重建。
- [同 scene CAV 共享 RSU Radar/BS-GPS] -> 按用户指定的 `(scene_id,cav_id)` 保持 assignment，并在 manifest/report 中作为 diagnostic-only overlap 完整披露。
- [真实 SNR 不可得] -> metadata 明确 `snr_available=false`，encoder 的 SNR 入口可空且不影响 feature；禁止用随机或常数 SNR 填补。
- [31-mask schedule 或诊断按 batch 顺序偏置] -> 使用 global sample position/epoch 确定性轮转并记录计数，样本级 paired 表按稳定 identity 聚合。
- [清理历史 source 时误删 PCPF sparse-CSI 原语] -> 删除前执行 import/reference 审计；dataset、outputs、cache、日志和 checkpoint 不读取、不移动、不删除。
- [逐 worker 重算 channel SHA 和打开大量小 cache 文件导致 CPU 饱和、GPU 饥饿] -> cache scan 发布协议/SHA 绑定的 packed CSI bundle，正式 dataset 只允许严格 bundle 命中；Camera/LiDAR/GPS 同时绑定已验证的严格缓存。
- [batch 64 超出目标 GPU 显存] -> 正式启动前执行真实 CUDA forward/backward smoke；若失败则生成新的显式 batch 配置并 fresh start，不修改或跨配置续用已有 checkpoint。
- [长阶段之间需要人工值守或误用半成品 checkpoint] -> 显式续跑器只接受完整 run status、达到预算的 last checkpoint 和 validation-best publication；任一步失败即停止且不创建后续可训练配置。

## Migration Plan

1. 合入新 capability/delta specs、模型/loss/config parser 和 focused tests；不改 U0 recipe。
2. 先以 synthetic tensor 完成三阶段冻结、checkpoint metadata、数值和 backward smoke。
3. 由本地 resolver 注入 block protocol、split audit、当前 manifest 的 train-only GPS scaler、fresh-start Stage 1 和后续 gate sidecar，生成 `outputs/` 下 resolved configs。
4. 先扫描 block train/validation 补齐固定 2x2 CSI cache并发布 packed bundle，绑定严格 RGB/LiDAR/GPS cache，再运行真实 MMW 一个 batch的 Stage 1/2/3 smoke；默认不构建 test loader。
5. 用户已显式授权在固定 `split_seed=0` 上启动 train seed 1/2/3 的拓扑监督 2x2 长训练；运行保持 test 封存、独立 GPU/输出目录且不自动重试。
6. 回滚只需停止使用 PCPF local configs/registry type；stable baseline 无迁移步骤。
7. 将其他 change 以 closure note 和仓库外快照归档；删除非 PCPF 本地实验 owner。
