## Context

新主线只包含两个可分离创新：训练期的 beam-topology prototype supervision，以及推理期的低开销 TBCP-3（2+1）finite probing。sensing 模型不读取 CSI/channel/beam power；probing policy 不更新模型。

## Architecture

```text
image/radar/gps/lidar
        -> independent encoders
        -> one shared temporal transformer
        -> one shared 64-beam prototype bank
        -> per-modality features query the shared prototype bank
        -> masked concat(features, availability)
        -> standard two-layer feature-fusion MLP
        -> fused feature queries the same prototype bank
        -> p_sense[64]
        -> stateless circular posterior statistics
        -> TBCP-3 requested-measurement policy
```

## Decisions

### 1. 单阶段、严格四模态

registry id 固定为 `four_modal_topology_predictor`。模型只接受 canonical 四模态和 temporal/availability masks；不存在 CSI 参数、risk/evidence、样本级 Router、`training_stage` 或旧 fusion mode。任一旧字段或旧 model id 必须在 config load 时失败。

### 2. posterior fusion 与静态可靠性修复

每个可用模态通过同一个 prototype bank 产生 64 类 probability。默认 control 的 `p_sense` 是可用模态 probability 的 arithmetic mean。无界诊断分支只新增四个样本无关的 trainable logits，经 availability mask 后 softmax 得到静态模态可靠性权重；初始值严格退化为 arithmetic mean。其 seed1 pilot 出现 LiDAR winner-take-all，因此正式修复分支在 softmax 前固定使用 `tanh(logit)`，把每个有效 logit 限制在 `[-1,1]`。四模态 Full mask 下任一权重最多为 `exp(2)/(exp(2)+3)≈71.1%`，从结构上阻止 97% 单模态塌缩，且不新增需要 validation 选择的正则系数。缺失模态严格为零，Single mask 在所有分支中都退化为对应单模态 probability。

该修复不是样本级动态融合：权重不读取当前 feature、posterior entropy、weather、domain、GT 或 RF measurement。它只由 train loss 更新并进入 checkpoint state dict；validation/test 只读。mean、无界诊断和 bounded static reliability 使用独立 fresh checkpoints，禁止把 validation 反事实直接写入权重，也禁止改变旧无界 checkpoint 的参数化语义。

### 3. 唯一训练 loss

单次训练同时优化 fused hard CE、availability-normalized unimodal hard CE、可选环形 soft CE 与 prototype alignment。topology-off 只将 soft/prototype 项置零；模型、数据、预算、mask schedule 与 seed 其余完全一致。

### 4. 原生 15-mask evidence

evaluator 对同一 validation-best checkpoint 直接遍历四模态 15 个非空 availability mask并保存 `fused_probability`。evidence schema 的 modalities 必须恰好为 `[image, radar, gps, lidar]`；五模态或 31-mask evidence 必须拒绝。

### 5. TBCP 保持独立

train-only topology likelihood、joint relative-dB update、expected-terminal-gain acquisition、K=3、2+1 batch feedback、requested-only simulator、synthetic measurement-error 与 defensive ablations保持既有数学定义。likelihood artifact 不进入模型 state dict，也不构成训练 stage。K=5/7/9 只作为固定预算敏感性，不能根据 validation 结果隐去。

### 6. 断代删除

删除旧源码、tracked templates、tests、docs 和确认清单中的 ignored outputs；不建立 alias、migration loader、旧 checkpoint converter 或 archive copy。保留正式 MMW split、train-only likelihood、ULA-DFT topology audit 和通用四模态数据缓存。

## Training and Ablation

- topology `{off,on}` × train seed `{1,2,3}`，共六个 fresh single-stage runs。
- 相同 `mmw_id_stratified_block_v1` seed 0、epochs、batch、workers、preprocessing、mask sampling 和 validation selection。
- 每个 checkpoint 生成原生 15-mask evidence，再运行 Direct / Posterior Top-3 / TBCP-3 2×3 nested ablation；TBCP-5/7/9 只进入预算敏感性表。
- 初始清理阶段不启动长训练；后续第 9/10 节实验均使用独立 ignored output、fresh-start 与封存 test。

### 6. 两个创新点的联动诊断

当前证据显示 topology-on 提升 Posterior Top-3，但在 K=5/7 时 TBCP 增益被强 RF 反馈部分吞掉，且 Radar-only 明显退化。主线先用 K=3 保留 sensing prior 的作用；不得用 validation 后验调参制造正交互。下一轮若仍需修复，优先做 train-only posterior reliability/entropy audit，再考虑 availability-aware topology loss 重训；不得把 validation/test 统计用于 loss 权重或温度选择。

### 7. 公平 ablation 与 baseline panel

主比较固定信息权限为 `S={image,radar,gps,lidar}`，不提供历史 beam index，也不提供当前 beam-power measurement。AMBER-Full 与 RMBP-MM 在本仓库中已经是 sensing-only local adaptation；不重复实现历史 beam 分支，也不把外部论文的 history-assisted 结果混入严格排序。

Topology 组件采用四格设计：hard CE、neighbor soft-only、prototype-only、soft+prototype full。另设普通 uniform label smoothing control，epsilon 预注册且不使用 validation 选值。Soft/prototype/uniform 条件共享 radar-robust whole-modality schedule、optimizer、epoch、seed 和 validation-best 选择。

对每个 baseline 和 topology predictor，统一保存 15-mask posterior evidence。Direct、Posterior Top-3、OpenLoopGain-3 与 TBCP-3 共享同一个 train-only likelihood、K=3、requested-only simulator 和 validation identity；baseline evidence 通过独立 generic adapter 生成，不放宽 native topology checkpoint loader。

历史 beam 若需展示，只能作为单独 privileged reference（Last-beam copy/Train-only Markov），不得与 sensing-only 主表混列。主表分别报告 predictor effect、同预算 TBCP 增益和 system-level `S+P3` 对照。

### 8. Radar robustness evidence

Radar-robust whole-modality on/off 三 seed 重训已完成：topology-on 在 All-15 与主要 sensing mask 上改善，但 Radar-only 仍是负交互。该结果作为边界披露，不再继续 validation-driven 调参；完整证据保留在独立 ignored output 根目录。

### 9. Full-posterior reliability repair

冻结 validation evidence 的诊断显示，Full mask 下 topology-on Direct Top-1 高于两个 sensing-only baseline，但固定四模态等权平均使近随机 Radar posterior 占 25%，导致 Posterior Top-3 与 TBCP-3 落后；同 checkpoint 去除 Radar 的反事实显著恢复 Top-3。修复因此限定为上述四标量 static reliability，而不是温度缩放或样本级 Router。温度缩放不改变 beam 排序，不能解决该问题。

实验先运行 static-reliability topology-on seed1 pilot，并与既有 matched mean topology-on seed1 比较 Full、missing-radar、All-15、Drop-1/2、Single、Radar-only和四个权重。pilot 只作为 claim-ineligible go/no-go 诊断，不据此调权重或修改配置。只有 Posterior Top-3/TBCP-3 的目标缺口得到改善且总体 missing-modality 曲线无明显退化，才继续 `fusion={mean,static_reliability}` × `topology={off,on}` × seed `{1,2,3}` 的完整 matched panel。所有运行 fresh-start、validation-only、test sealed。

seed1 pilot 使用 epoch10 validation-best checkpoint，学习到 `[image,radar,gps,lidar]=[16.39%,0.36%,0.48%,82.77%]`。相对 matched mean seed1，Full Posterior Top-3/TBCP-3 分别提升 `+6.32/+4.00 pp`，All-15 提升 `+4.41/+2.54 pp`，说明弱 Radar 等权稀释诊断成立；但 GPS-only TBCP-3 下降 `5.21 pp`，且 last checkpoint 的 LiDAR 权重继续升至 `97.08%`，表明无约束 static logits 存在 winner-take-all collapse。该证据保持 validation-only、claim-ineligible，不触发无界分支的完整多 seed panel。下一步只运行固定 `tanh` 边界的 bounded static reliability topology-on seed1 fresh pilot；边界在看到该 pilot 结果前预注册，不使用 validation 调节。

bounded seed1 pilot 最终权重为 `[43.95%,6.02%,6.04%,43.99%]`，结构上阻止单模态97%独占，但 raw logits 全部进入 `tanh` 饱和区，形成 Image/LiDAR 对 Radar/GPS 的两组边界选择。相对 matched mean seed1，Full Direct/Posterior/TBCP-3 为 `+3.68/+2.26/+0.67 pp`，All-15 TBCP-3 为 `-0.01 pp`；因此不触发 bounded 多seed panel。

### 10. 标准 mask-aware feature fusion

取消 reliability gate 作为修复方向不等于禁止普通可学习backbone。新分支固定为 `masked_feature_mlp`：先将不可用 `h_m` 置零，再拼接四个64维模态特征与4维availability mask，通过 `LayerNorm -> Linear(260,128) -> GELU -> Dropout -> Linear(128,64)` 得到融合特征。它不产生四个标量权重，不读取posterior uncertainty、weather/domain、GT、CSI、历史beam或RF measurement，也不作为创新点。

全模型只有一个 `BeamPrototypeBank[64,64]`。每个单模态特征 `h_m` 直接查询该Bank并接受 availability-normalized hard CE 与 neighbor-soft CE；融合特征 `z_fused` 查询同一个Bank并接受 fused hard CE 与 fused topology-prototype supervision。`lambda_modality_proto` 在该pilot保持0，因为单模态neighbor-soft已经在同一Bank上执行相同聚类目标，重复开启会改变相对loss量级并重现弱Radar过约束。topology off/on未来必须共享完全相同的feature-fusion参数，仅切换topology supervision。

在用户明确授权整夜使用八张GPU后，固定运行 topology on/off × seed 1/2/3 六条matched fresh validation-only训练，并各运行soft-only/prototype-only seed1一条组件诊断。主结论只使用matched on/off三seed；两个单seed条件只用于判断neighbor-soft与fused prototype项的独立贡献。不得根据这八条validation结果修改MLP宽度、loss系数或mask schedule。

在后续公平性诊断中，用户明确授权将 masked-feature Prototype-only 从 seed1 组件筛选扩展到 seed2/3 稳定性复现。新增两条训练必须直接复用 seed1 的tracked template、whole-modality schedule、40 epoch、validation-best、数据protocol与预处理，只允许改变train seed和独立run/output名称；不得根据seed1 validation结果调整loss权重、模型或训练超参数。三seed仅用于报告Prototype-only的均值、标准差和paired ablation，不访问outer test。

Prototype-only 三seed复现完成后，All-15 的 Direct / Posterior Top-3 / TBCP-3 分别为 `59.49±0.30% / 78.91±0.06% / 82.48±0.11%`。相对 matched Hard control，TBCP-3 在 All-15 / Full / Drop-1 / Drop-2 / Single Macro 上分别提升 `+3.46 / +3.29 / +3.59 / +4.03 / +2.50 pp`；前三个seed的 All-15 paired delta 均为正。该结果确认 Prototype-only 在当前 masked-feature backbone 上具有稳定收益，但 Radar-only 仍是 Single Worst，TBCP-3 仅为 `25.61±1.82%`。全部证据保持 validation-only、claim-ineligible，outer test 未访问。

### 11. 统一创新点1为Joint Topology Prototype Loss

feature-fusion诊断显示soft-only与prototype-only seed1均优于两项相加，说明两项对同一共享Bank和同一环形soft target重复施压。正式创新点1因此不再把它们当成两个可叠加目标，而定义为：

`L_joint_topo = 0.5 * (L_topo(z_fused,y) + mean_available_m L_topo(h_m,y))`。

总loss固定为fused hard CE、availability-normalized unimodal hard CE，再加唯一 `joint_topology_weight * L_joint_topo`。新Joint模板将旧 `unimodal_soft_weight/lambda_proto/lambda_modality_proto` 全部置零，`use_beam_prototype_alignment=false`；这不会删除旧诊断能力，只避免正式主线重复计权。权重预注册为0.1，因为等权平均后其总量级与已运行成功的任一单项0.1一致，而不是两项相加；不根据后续validation结果调节。

先只运行Hard-CE与Joint-Topology seed1两条fresh matched训练。比较Full、All-15、Drop-1/2、Single、四个only模态的Direct/Posterior Top-3/TBCP-3；只有Joint相对Hard改善总体Posterior/TBCP且不复现组合负交互，才补三seed。

## Risks

### 12. DeepSense6G secondary transfer panel

DeepSense6G Scene31–34 只有兼容五帧输入的官方 train/test CSV，没有同schema validation。secondary panel固定使用已生成的 `deepsense6g_twc_secondary_v1` 过滤manifest：只剔除 `future_beam1` 不是恰好64个finite nonnegative值的窗口，保留pooled train/test `13240/4090`，不插值、不静默跳过。三种方法均固定40 epoch、batch32、seed 1/2/3和last checkpoint；test只在每条固定训练完成后执行一次，不参与epoch或超参数选择。

DeepSense6G没有MMW的ULA-DFT codebook audit。train-only功率审计表明内部相邻beam有局部平滑，但0/63边界不足以支持环邻接。因此 Prototype-only transfer使用 `linear_index_v1`：Gaussian neighbor target按普通绝对index距离计算，不连接0与63；不得携带ULA descriptor/audit。AMBER-Full-local和RMBP-MM-local复用各自现有encoder/core，三者共享四模态、五帧、64类、missing schedule和评测mask。

该面板只报告Direct Top-1/3/5、DBA、15-mask与按Scene/缺失模态数汇总，标记secondary exploratory/claim-ineligible。现有TBCP-3的train likelihood、modulo offset update和radio topology均绑定MMW ULA phase cycle；在DeepSense codebook topology完成独立审计前，不运行或宣称TBCP跨数据集泛化，也不把现有T2/U-Mask结果替代当前Prototype-only模型。

### 13. 冻结主线的防守统计与一次性 MMW test

Prototype-only、Hard control、RMBP-MM-local、AMBER-Full-local 的结构、训练超参数、validation-best checkpoint、train-only likelihood、TBCP-3策略与三seed集合在任何test访问前冻结。先完成validation上的三项只读分析：`sigma_db={0,3,6}` matched synthetic measurement-error replay（3/6 dB各三个replica，0 dB为确定性锚点）、trajectory/domain cluster paired bootstrap，以及参数量/FLOPs/单样本sensing前向计时。分析不得更新模型、likelihood或策略。

paired bootstrap以相同seed、stable sample、missing pattern和method逐项配对。主报告分别给出trajectory-cluster和domain-cluster的95% percentile CI；不得把`5931×15`行当成独立样本，也不得根据两种cluster口径选择有利结果。固定bootstrap seed为`20260813`、replicates为`10000`，比较Prototype-only相对Hard与RMBP的Direct、Posterior Top-3和TBCP-3配对提升。trajectory跨三个天气domain，二者属于crossed dependence；不得把`domain×trajectory`交叉单元伪装成额外独立物理轨迹。

MMW test只允许一个manifest驱动的final panel入口。入口必须在构建任何test dataset前验证恰好四方法×seed `{1,2,3}` 的12个validation-best checkpoint、resolved config SHA256、MMW protocol/topology/normalization、15-mask定义、likelihood与代码版本，并写入不可覆盖的seal manifest。全部检查通过后才可统一构建test并生成Direct、Posterior Top-3、TBCP-3及缺失0/1/2/3模态结果；不得因test结果修改模型、超参数、策略或选择seed。

复杂度报告固定使用batch=1、五帧完整四模态、预加载输入、A40、FP32、CUDA同步、20次warmup与100次测量，报告参数量、PyTorch profiler可识别operator FLOPs及forward median/p95。FLOPs必须标注为profiler-covered而非硬件能耗。TBCP-3系统开销固定为3/64 beam measurements、2个measurement rounds、1次controller feedback update；相对Full-64只可声称beam measurement count减少`61/64=95.3125%`，不得等价为实测毫秒时延降低。

- 删除 ignored outputs 不可恢复：执行前使用显式路径清单，保留 split/topology/calibration。
- 大范围删除可能误伤通用 trainer：删除前以引用搜索确认 owner，删除后运行 full pytest 与 compile。
- 旧 checkpoint 全部失效是预期行为，不通过放宽 strict load 解决。
