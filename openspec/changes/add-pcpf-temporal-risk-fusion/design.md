## Context

当前 U0 的四个 encoder 均以五帧序列为输入并返回 `[B,T,D]`，随后由 `UMaskBeamJEPA` 做 masked mean、构造单模态 prototype logits，再用 supervised Router 融合。训练缺失契约的真实形状是 `modality_temporal_mask=[B,T,4]`，`available_modalities=mask.any(dim=1)`；`BeamPrototypeBank` 持有唯一可学习 `[64,64]` prototype，使用归一化 feature/prototype 的 cosine logits，现有 `prototype_alignment_loss` 已覆盖 fused/modality feature、availability mask 和 topology soft target。

通用 trainer 在模型构造和 initialization checkpoint 加载后创建 optimizer，通过 training extension 替换 base loss，并按 inner validation loss 保存 `best.pth`。15-mask evaluator 已覆盖 Full、drop-1、drop-2、Single、Within-3、circular MAE、ECE 和宏/最差聚合；MMW pooled dataset metadata 提供 weather/domain。最新 CSI/TSPC-V2 是未注册的本地研究 workflow，读取独立 pilot/cache 并保持 Full/CSI-off 硬旁路；PCPF-T 不读取、导入或初始化其中任何 CSI owner。

历史工作树曾同时存在多个 active change。此次收敛将它们按真实完成/停止状态保存到仓库外快照，当前工作树只保留 PCPF-T change；CSI/TSPC、trajectory、U0 与保留 baseline 的 source 不因 change 归档而删除。

## Goals / Non-Goals

**Goals:**

- 在 canonical `image, radar, gps, lidar` 和五帧 mask contract 上实现独立注册的 PCPF-T。
- 让每个模态先独立经过共享 Temporal Transformer，再查询同一个 `BeamPrototypeBank`。
- 以低自由度、非负系数公式学习 `U_var/U_proto/U_temp/U_conflict` 到 topology risk 的映射，并用固定 precision-style 公式融合概率。
- 复用通用 optimizer、training extension、checkpoint、validation-best、MMW protocol 和 15-mask metrics。
- 对训练阶段、train-only 拟合状态、checkpoint 来源角色、Stage 2 gate、FP32 风险计算和 missing 权重做 fail-closed 验证。
- 提供无需 outer test 或长训练即可执行的 synthetic/one-batch smoke 和 resolved-config workflow。

**Non-Goals:**

- 不改变、替换或删除 U0 Router，也不把 PCPF-T 参数加入 U0。
- 不把 PCPF-T 变成第四个 canonical MMW recipe或新增 public CLI。
- 不读取 channel、CSI、path、beam power、历史 beam、天气、场景或 corruption metadata 作为模型输入/target。
- 不实现 frame-level supervised CE；数据没有独立逐帧未来标签时，不用同一个未来标签伪造五个目标。
- 不默认加入模态间直接对比、跨模态 attention、feature concat 或任意四维动态 Router。
- 本 change 不运行 outer test、multi-seed、整夜训练，也不据 validation/test 结果事后修改 Stage 2 gate。

## Decisions

### 1. 独立整模型而非继承 UMaskBeamJEPA

新增 `pcpf_temporal_risk_fusion`，复用 registry encoder 配置和 `BeamPrototypeBank`，但不继承 `UMaskBeamJEPA`。继承会无条件创建 reliability head、classifier 和 supervised Router，既污染 checkpoint，也使“解析式动态融合无自由 Router”无法审计。公共 import 只注册 class；只有 recipe 显式声明该 type 时才实例化参数。

替代方案是继续给 U0 增加 opt-in 分支；这会扩大 U0 forward 和 state dict，拒绝采用。

### 2. 一个共享 Temporal Transformer，逐模态独立进入

`SharedTemporalTransformer` 接受 `[B,T,4,D]` 与 `[B,T,4]`，为每个模态应用独立 input LayerNorm/可选线性 adapter，再将张量重排为 `[B*4,T,D]`。共享 time embedding `[T,D]`、modality embedding `[4,D]`、T-CLS 和两层 `TransformerEncoder`；T-CLS 永远可见，frame padding mask 来自真实 temporal mask，`causal=false`。

输出 frame token `[B,T,4,D]`、CLS `[B,4,D]`、valid fraction 和 availability。整模态缺失时允许 Transformer 完成计算，但随后将该模态的 frame/CLS 显式置零，防止 T-CLS 或 embedding 产生伪证据。风险估计前没有跨模态 attention 或 feature concat；唯一跨模态量是基于 detached/显式 probability 的弱 conflict 标量。

### 3. 单一 prototype bank 同时承担专家预测和语义对齐

模型只创建一个 `BeamPrototypeBank`。Stage 1 的 deterministic unimodal logits 来自 temporal CLS；`prototype_alignment_loss` 直接复用 fused/modality soft topology alignment，不复制 InfoNCE。新增的 unimodal loss 对每个样本的可用模态先算 hard CE 与 soft topology CE，再按该样本可用模态数归一化。

Stage 1 默认 uniform 概率融合；static learnable prior 仅作为显式 control。概率融合结果以 `log(p_fused)` 暴露为 canonical logits，避免把概率加权误写成 logits 加权。

### 4. 共享概率嵌入和显式四项风险

Stage 2 使用一个共享 `ProbabilityEmbeddingHead`：零初始化 `DeltaMu` 末层使 `mu=h`，零权重且 bias=-4 的 logvar head 产生 `[B,4,D]`，并 clamp 到 `[-8,4]`。训练可用重参数样本仅服务 preserve loss；deterministic `mu` logits 始终用于 `R_star`、风险与评估。

四项风险全部在 autocast disabled 的 FP32 block 中计算：

- `U_var = mean(exp(logvar))`；
- `U_proto = 1 - max cosine(mu, prototype)`；
- `U_temp` 从有效 frame prototype distribution 的 circular mean 得到角度序列，至少三帧时先用最短 circular difference 展开，再拟合一阶趋势并以 `pi^2` 归一化残差；不足三帧为零且 `temp_valid=false`；
- `U_conflict` 为当前模态与其他可用模态均值分布的 JS divergence；Single 时为零。

`raw_risk=softplus(sum softplus(rho_x)*normalized(U_x)+bias)`，四个系数默认跨模态共享。禁用消融分量时在公式入口置零且不改变其余分量定义。

### 5. train-only 状态通过 opt-in stage preparation 拟合

通用 trainer 在 initialization checkpoint 加载后、optimizer 创建前，仅当模型实现 `prepare_training_stage(...)` 时调用该方法。PCPF-T 用 train dataset 的顺序只读 loader：Stage 2 拟合并冻结四项风险的 mean/std；Stage 3 以 deterministic detached unimodal probability 和真实未来 label 计算每模态 `mean_train_risk`。该 pass 不读取 validation/test，保存 JSON 与 model buffers，并恢复原 loader RNG 状态。

Stage 2 初始 logvar 恒为 -4，因此 `U_var` 在 preparation 时退化为常数。`risk.normalization_epsilon` 是风险分量的标准差下限而非机器精度 epsilon，正式配置固定为 `0.01`；拟合标准差低于该值时保存并使用下限，避免随后训练 probability head 时把微小 `U_var` 变化放大到 Softplus 饱和区。

smoke 可显式限制 preparation batches；正式配置必须为全 train pass。U0 和保留 baseline 没有该方法，所以不会多走 forward、修改 RNG 或改变 optimizer。

替代方案是在每个 Stage 2 batch 在线更新统计；这会让 normalization 与风险头共同漂移且 validation 时点难以复现，拒绝采用。

### 6. 固定的三阶段冻结和 fail-closed checkpoint 来源

模型构造时按 stage 设置 `requires_grad`，因此 optimizer 只看见允许参数；`train()` 会让全部冻结 module 保持 eval。启动 metadata 输出完整 trainable parameter names/count，extension 再断言实际集合等于预期集合：

| Stage | 训练参数 | 冻结参数 |
| --- | --- | --- |
| `stage1_expert` | encoders、encoder projections、temporal input adapter/共享 Transformer、prototype bank；static control 时仅额外 global prior | probability/risk、temperature/tau/eta、direct Router |
| `stage2_risk` | probability embedding、risk coefficients/bias | encoders、projection、Temporal Transformer、prototype、static prior、temperature/tau/eta、Router |
| `stage3_fusion` | 每模态 temperature、tau、可选 eta；显式配置时才微调共享 risk coefficients | expert/prototype/probability head，其余 control |
| `stage3b_optional_finetune` | Stage 3 scalars、probability/risk heads | encoders、projection、Temporal Transformer、prototype |

checkpoint payload/sidecar 增加通用 `model_metadata`，PCPF-T 记录 stage、fusion mode、train-only stats、claim eligibility 和 gate 状态。初始化请求可声明 `expected_source_training_stage`；Stage 2/3/3B 分别只接受 Stage 1/2/3A 的 `validation_best`，不匹配时在加载 optimizer 前失败。通用 `best.pth` 保持不变，并为 PCPF-T 创建 `stage1_best.pth` 等只读相对 symlink，兼顾现有工具和研究命名。

### 7. Stage 2 gate 是独立只读评估产物

Stage 2 loss 使用 detached `R_star=sum p_mu(k)*normalized_topology_distance(k,y)`、masked Huber、margin pair ranking、很小 Gaussian KL 和 sampled/deterministic preserve topology CE；不使用 fused CE 更新风险。

评估器从同一 checkpoint 输出 overall、modality、weather、domain、mask group 的 Pearson/Spearman、decile/calibration 和 confident-but-wrong 诊断。门槛来自 tracked config，报告只计算 pass/fail 和失败原因，绝不写回阈值。Stage 3 resolved config 必须绑定 gate JSON 与 SHA256 且 `stage2_gate_passed=true`；launcher 不自动串联启动 Stage 3。

### 8. Stage 3 只做概率校准和解析融合

模型保存 train-only `mean_train_risk_m`。`eta` 为非负标量，`a_m=exp(-eta*mean_train_risk_m)`；`T_m=Tmin+softplus(t_m)`，`tau=tau_min+softplus(raw_tau)`。在 FP32 中用 log-score `log(a_m)-risk/tau` 加 availability masked softmax，避免 `exp` 下溢，并保证 missing weight 为零、Single weight 为一、行和为一。

`p_fused=sum w_m*softmax(logits_m/T_m)`。Stage 3A loss 为 NLL 加可选 topology soft CE；Stage 3B 额外保持风险 Huber/ranking，监控 Spearman 的回退由本地 runner 根据 Stage 3A checkpoint 执行，不将 validation 结果输入 forward。

### 9. 对照共享 expert 初始化，但不污染主模型

Uniform、Static Prior、PCPF-T 和 `no-*` 直接通过解析 fusion mode/分量开关替换。Direct Router 是仅在 A2 control config 中创建的旧式标量特征 MLP，metadata 明确 `control_only=true`；A4 主模型没有该 module。CUAF 无法由任务简称唯一定位逐项公式，因此实现只命名为 `cuaf_local_adaptation`，使用 normalized entropy、margin risk 与 JS conflict 的固定解析组合，不宣称论文复现。

动态替换评估缓存同一次 A4 forward 的 unimodal logits 与风险分量，再应用 A0--A3 各自 validation-best checkpoint 中已训练的 temperature、tau 或 Router 参数；不得为 control 重跑 encoder。所有 control checkpoint 必须与 A4 具有相同 Stage 1 expert fingerprint，且路径、SHA256、fusion mode 和 checkpoint role 写入 matrix provenance，否则拒绝比较。

### 10. 本地配置和评估面，不扩大 canonical surface

tracked template 位于 `tools/configs/pcpf/`，通过现有 `_base_`、config loader、MMW protocol 注入和严格 PCPF parser 解析；不在 `configs/mmw/` 增加 canonical recipe。一个 `tools/run_pcpf.py` 负责 resolve/preflight/train/smoke，一个 `tools/eval_pcpf.py` 负责 gate/15-mask/替换/诊断；不复制 trainer。所有 resolved config、stats、reports、checkpoint 和 smoke 输出位于 `outputs/pcpf_temporal_risk/`。

## Risks / Trade-offs

- [Stage 2 risk 可能与 `R_star` 无相关性或常数化] -> 预注册 gate 失败即停止，不自动启动 Stage 3，也不修改阈值。
- [Stage 2 初始 `U_var` 方差为零] -> 使用预注册的 `0.01` normalization std floor，并以 preparation 后单优化步的有限梯度和非零风险回归测试防止塌缩。
- [Transformer 对全缺失模态仍能由 CLS 产生非零值] -> forward 后按 availability 显式清零并测试 probability/risk/weight。
- [FP16/BF16 下 exp/log/JS/KL 下溢] -> 风险、校准和融合统一在 FP32 中执行，输出结束再 cast。
- [train-only preparation pass 增加启动耗时] -> 正式运行完整遍历以保证统计契约；仅 smoke 允许限 batch，并在 metadata 标记不具 claim 资格。
- [Stage 1 随机、未训练 probability/risk 参数进入 checkpoint] -> 它们存在于 PCPF-T state 以支持严格 stage 初始化，但被冻结且不参与 forward/loss；checkpoint metadata 明确 inactive modules。U0 state dict 不受影响。
- [已有 checkpoint 缺少 `model_metadata`] -> 只有声明 `expected_source_training_stage` 的 PCPF 初始化要求该字段；旧 U0 加载路径继续接受现有 schema。
- [历史 development test 已用于开发] -> 所有 PCPF 输出固定 `claim_ineligible=true`，outer test loader 默认不构建。
- [清理历史 change 时误删仍被 CSI/TSPC 使用的共享模块] -> 删除前执行 import/reference 审计，只移除无保留路线引用的失败叶子；dataset、outputs、cache、日志和 checkpoint 不读取、不移动、不删除。

## Migration Plan

1. 合入新 capability/delta specs、模型/loss/config parser 和 focused tests；不改 U0 recipe。
2. 先以 synthetic tensor 完成四阶段冻结、checkpoint metadata、数值和 backward smoke。
3. 由本地 resolver 注入 clean protocol、audit、Stage 1 initialization 和 gate sidecar，生成 `outputs/` 下 resolved configs。
4. 只运行真实 MMW 一个 batch 的 Stage 1/2/3 smoke；不自动开始长训练。
5. 正式实验由用户依次显式启动 Stage 1、Stage 2、gate、Stage 3A；Stage 3B 仅在 A4 超过 A1 且风险相关性未退化时运行。
6. 回滚只需停止使用 PCPF local configs/registry type；U0、CSI/TSPC 和现有 checkpoint 无迁移步骤。
7. 将其他 change 以 closure note 和仓库外快照归档；只删除已有明确停止结论且无反向依赖的实验 owner。

## Open Questions

- 当前 source recipe 只携带 cyclic topology，物理 ULA-DFT topology provenance 位于本地产物。正式 resolved config 必须绑定哪一份已审计 topology descriptor，由运行时 protocol 决定；源码测试使用等价的 64 类 cycle，不读取 `outputs/`。
- Stage 3B 的“Spearman 明显下降”数值阈值需在正式运行前随实验 protocol 预注册；本实现只提供配置字段和 fail-closed 回退，不替用户猜测阈值。
