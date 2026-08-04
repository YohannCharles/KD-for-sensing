## 1. 边界与配置契约

- [x] 1.1 新增严格 PCPF-T model/loss 配置解析与 stage、维度、非负系数、禁止输入校验
- [x] 1.2 注册独立 `pcpf_temporal_risk_fusion`，确认 U0 构造与 state dict 不包含 PCPF owner
- [x] 1.3 新增 `tools/configs/pcpf/` 的 base、三阶段、A0--A4 和分量消融模板，不修改 canonical MMW recipe

## 2. Temporal Expert 与 Prototype

- [x] 2.1 实现唯一共享的逐模态 Temporal Transformer、真实 `[B,T,4]` padding mask 和全缺失清零
- [x] 2.2 实现四 encoder、projection、唯一 `BeamPrototypeBank` 与 deterministic unimodal probability forward
- [x] 2.3 实现 availability-aware unimodal hard/soft topology loss，并复用现有 prototype alignment

## 3. 概率嵌入与拓扑风险

- [x] 3.1 实现共享 ProbabilityEmbeddingHead、零初始化 identity 行为与训练采样/eval deterministic 路径
- [x] 3.2 实现 FP32 `U_var`、`U_proto`、circular `U_temp`、JS `U_conflict` 和 disabled-component 消融
- [x] 3.3 实现 detached `R_star`、masked Huber、pair ranking、Gaussian KL 与 topology preserve loss
- [x] 3.4 实现只读 train-only normalization 与 `mean_train_risk` stage preparation、buffer 和 provenance

## 4. 解析融合与阶段训练

- [x] 4.1 实现 uniform、static prior、analytic PCPF、direct Router control 和 `cuaf_local_adaptation` 概率融合
- [x] 4.2 实现 FP32 temperature、static capability、risk/tau log-score 与 missing/Single 权重约束
- [x] 4.3 实现三个 stage 的精确冻结、frozen module eval、trainable name/count metadata 和断言
- [x] 4.4 接入共享 trainer 的 opt-in preparation、stage-specific loss 和 validation loss，保持 U0 默认路径不变
- [x] 4.5 扩展 checkpoint model metadata、source-stage fail-closed 校验和 PCPF validation-best stage alias

## 5. 评估与本地工具

- [x] 5.1 实现 Stage 2 gate 的相关性、分位、常数化、天气/domain/mask-group 报告和固定阈值判定
- [x] 5.2 扩展 15-mask 评估输出 weight/risk/calibration/temperature/claim provenance 诊断
- [x] 5.3 实现 `tools/run_pcpf.py` 的 resolve/preflight/synthetic/one-batch smoke 与 `tools/eval_pcpf.py` 的 gate/matrix 入口
- [x] 5.4 对所有 Stage 3 训练请求校验 gate JSON、SHA256、通过状态与 Stage 1 expert fingerprint

## 6. 测试与验证

- [x] 6.1 使用 `conda run -n kd_mm_beam pytest` 覆盖 temporal shape/mask、四项风险、概率融合、stage freezing 和数值稳定性
- [x] 6.2 使用 `conda run -n kd_mm_beam pytest` 覆盖配置拒绝、checkpoint 来源、gate、15-mask 聚合和 U0 隔离
- [x] 6.3 使用 `conda run -n kd_mm_beam` 完成 synthetic 三阶段 forward/backward smoke 并记录 shape、loss、梯度、权重和 NaN/Inf
- [x] 6.4 使用 `conda run -n kd_mm_beam` 完成真实 MMW 单 batch Stage 1/2/3 smoke，不访问 outer test、不启动长训练
- [x] 6.5 运行 `openspec validate add-pcpf-temporal-risk-fusion --strict`、`make verify-quick` 与相关 focused tests，记录既有 active-change 风险

## 7. 主线收敛与历史归档

- [x] 7.1 为三个历史 active change 写入真实 closure note，并保存仓库外快照；不把未完成任务伪装为完成
- [x] 7.2 从当前工作树移除已停止且孤立的 probabilistic prototype owner
- [x] 7.3 保留并验证 U0、AMBER-Full、RMBP-MM、DeepSense6G、当前 trajectory protocol 与全部 cache 边界
- [x] 7.4 更新 current specs、README、agent navigation、maintainer context 和 architecture boundary，使 PCPF-T 成为唯一 active research mainline
- [x] 7.5 成组移除已停止且无保留路线反向依赖的 SMSL R5、CPSU、Router observability/conformal 与 BTMA closure owner，并删除根目录原始 PCPF 提示和汇报 PDF

## 8. Stage 2 数值塌缩修复

- [x] 8.1 将风险 normalization std floor 固定为 `0.01`，同步正式配置与模型默认值
- [x] 8.2 增加退化 `U_var` preparation 后单优化步回归测试，并完成 focused/OpenSpec 验证
- [x] 8.3 补齐 15-mask matrix 的 checkpoint、数据 protocol、gate 与 normalization provenance，并增加聚焦回归测试
- [x] 8.4 让 A0--A3 control 在同一次 A4 forward 上应用各自 validation-best 参数，记录 control provenance 并完成 seed1 公平矩阵

## 9. 历史 sparse CSI 数据契约

- [x] 9.1 审计 PCPF-T sparse-CSI、旧协议污染风险、历史 channel 时间顺序、固定 2x2 selection、codebook/checkpoint identity，并记录 `docs/pcpf_sparse_csi/source_audit.md`
- [x] 9.2 实现 trajectory 样本自身五帧历史 channel 到 `[5,2,2]` complex sparse CSI 的确定性 sidecar/cache，写出 selection/codebook/frequency/history provenance
- [x] 9.3 让 SparsePilotEncoder 在真实 SNR 缺失时显式工作并输出 `snr_available=false`；禁止 AWGN、dropout、corruption 和伪造 SNR
- [x] 9.4 将 sparse-CSI 数据绑定迁移到当时的 `mmw_trajectory_disjoint` seed manifest；该绑定现已由第 20 节的新协议替代
- [x] 9.5 在训练前扫描 trajectory train/validation 补齐固定 2x2 CSI cache，保存协议、样本数、唯一 channel 数与 `outer_test_accessed=false` 清单

## 10. 五模态模型与初始化

- [x] 10.1 为 PCPF-T 增加 `use_sparse_csi` opt-in，默认四模态不创建 CSI owner
- [x] 10.2 实现 CSI temporal encoder/projection、五模态共享 Transformer/prototype/probability/risk/fusion 与完全缺失硬置零
- [x] 10.3 删除四转五 checkpoint 迁移，trajectory sparse-CSI Stage 1 只允许 fresh start
- [x] 10.4 实现确定性等频 31-subset 训练 schedule、逐 epoch mask 计数和相同缺失 mask 的五路传播
- [x] 10.5 增加本地 sparse-CSI Stage 1/2/3、R0--R7 与 gate 配置，不扩大 canonical recipe 或公共 CLI
- [x] 10.6 禁止用任何旧 split checkpoint 初始化，允许 trajectory sparse-CSI Stage 1 fresh start，并对后续 checkpoint 保持同协议校验

## 11. 五模态评估与机制诊断

- [x] 11.1 将 evaluator 泛化为默认 15/opt-in 31 个非空 mask，并约束 R0--R7 使用同一 split/seed/budget/expert fingerprint
- [x] 11.2 保存可复算样本级表：identity/group、label/prediction、每专家 probability/risk/weight/R_star/confidence/correctness、availability 与 CSI 质量字段
- [x] 11.3 实现 D0 原始 risk、D1 domain+mask 内 risk shuffle、D2 domain+mask 平均 risk、D3 static prior，以及独立的 risk/component/error/weight 诊断和按 group 的 paired bootstrap

## 12. 回归与 smoke

- [x] 12.1 使用 `conda run -n kd_mm_beam pytest` 覆盖固定 CSI selection/复数/SNR 缺失、五模态 shape/mask/risk/weight、31-subset 等频和 fresh-start 约束
- [x] 12.2 覆盖 time-order/split/resource 泄漏拒绝、禁止输入/噪声拒绝、四模态 checkpoint/forward 数值回归与 U0/CSI-TSPC 隔离
- [x] 12.3 完成 synthetic 与真实 MMW 单 batch Stage 1/2/3 smoke，记录 shape、loss、梯度、mask 计数、NaN/Inf 和 GPU peak memory
- [x] 12.4 在当时绑定的 trajectory train/validation 上完成 fresh-start 五模态真实单 batch Stage 1/2/3 CUDA smoke，并核验同协议 checkpoint provenance

## 13. 主线源码收缩

- [x] 13.1 删除非 PCPF 本地实验 runner、配置、脚本、专用源码、文档和测试
- [x] 13.2 删除 Stage 3B、公式候选筛选和四转五 checkpoint 迁移
- [x] 13.3 删除未启用的 TensorBoard/Matplotlib、参数组 DSL 与冗余配置/目标包装层
- [x] 13.4 更新 current specs、README、维护导航并完成 focused、compile 与 strict OpenSpec 验证

## 14. trajectory 训练吞吐修复

- [x] 14.1 为 train/validation sparse CSI cache scan 增加协议和 SHA256 绑定的 packed 2x2 bundle，并让正式 dataset 严格内存命中
- [x] 14.2 在 resolver 中恢复 RGB/LiDAR/GPS 严格缓存绑定，将 sparse-CSI 默认 batch 调整为 64、workers 调整为 8
- [x] 14.3 增加 packed cache、配置绑定和 fail-closed 回归测试，完成 focused、compile 与 strict OpenSpec 验证
- [x] 14.4 重建 trajectory cache bundle，完成真实 CUDA batch 64 显存/吞吐 smoke，并在新目录 fresh start Stage 1

## 15. trajectory 证据链纠偏与真实 R0

- [x] 15.1 在 proposal 与 `mmw-trajectory-disjoint-protocol` delta 中声明 channel 默认仅用于审计，并将严格同样本历史 sparse CSI 定义为 PCPF-T 唯一 opt-in 例外；在 design 中收敛 topology 证据边界并完成 change strict validation
- [x] 15.2 修正 validation/trajectory provenance：从绑定的 `data_protocol` 写出真实 train/validation role、protocol fingerprint、validation identity/count/order/seed 与 `outer_test_accessed=false`，并让 pooled validation、Stage 2 gate 和 R0--R7 matrix 拒绝 `splits.test` 或硬编码的旧 role
- [x] 15.3 撤销旧 `mmw_trajectory_disjoint` 32,235/14,625 上的四模态真实 R0 长训练；该协议已失效且本轮明确禁止长训练，后续 R0 必须等待第 20 节新 manifest
- [x] 15.4 落实经审计的 ULA-DFT topology descriptor：以稳定 ID/SHA256 绑定 tracked/resolved config、checkpoint、gate 与 matrix provenance，并在缺失或不匹配时失败关闭；当前 `cyclic_index_v1` 运行保持 `claim_ineligible` 且不得进入正式 R0--R7 或归档证据
- [x] 15.5 增加 pooled validation、trajectory gate、真实 R0、sparse-CSI/protocol metadata 的 focused 回归测试，并运行相关 `conda run -n kd_mm_beam pytest`、`make verify-quick`、`make verify-cli-config` 与 `make verify-compile`
- [x] 15.6 运行 `openspec validate --all --strict`、`make verify-full` 和 `conda run -n kd_mm_beam pytest -q`，核对工作树不包含本地数据、运行产物、日志或 checkpoint 后再评估归档

## 16. trajectory 数据画像与改进证据

- [x] 16.1 在 proposal、design 与 trajectory delta spec 中定义 train/validation-only dataset audit、diagnostic-only 字段、group-aware 统计和 ignored artifact 边界
- [x] 16.2 实现 protocol/audit/hash fail-closed 的本地分析工具，覆盖组成、跨天气轨迹内容配对、标签、时序、几何、beam-power、四模态与 sparse-CSI 质量/shift
- [x] 16.3 实现唯一资源去重、跨 split 配对签名重合、确定性时序签名和固定预算 trajectory-group/scenario-LOO/weather-LOO diagnostic probe，不使用 validation 选择分析配置
- [x] 16.4 增加 sealed-test 不可读、hash/count 拒绝、circular transition、train-only 拟合、确定性输出与坏资源定位测试
- [x] 16.5 对当时绑定的 trajectory split 全量运行，生成机器可读表、图表和引用证据的中文改进优先级报告，核验 `outer_test_accessed=false`；协议迁移后旧报告仅作历史开发记录
- [x] 16.6 运行 focused tests、`openspec validate add-pcpf-temporal-risk-fusion --strict`、`make verify-quick` 与 `make verify-compile`

## 17. 三阶段自动续跑

- [x] 17.1 在 proposal、design 与 capability spec 中定义显式续跑、完整 checkpoint、独立进程和 Stage 2 gate 失败关闭契约
- [x] 17.2 在 `tools/run_pcpf.py` 实现等待当前 Stage 1、解析/训练 Stage 2、运行无界 gate、通过后解析/训练 Stage 3 的最小本地编排
- [x] 17.3 增加完成状态、epoch、checkpoint role/stage 和运行名推导的聚焦测试，并运行配置、CLI、compile 与 strict OpenSpec 验证

## 18. MMW 唯一 trajectory-disjoint seed 0 协议

- [x] 18.1 将 current/change specs 更新为 `(scene_id,cav_id)`、三天气绑定、surplus CAV 固定 train、默认 split seed 0 与 test 封存契约
- [x] 18.2 替换旧资源连通 80/10/10 builder，实现当时的 11/2/3 协议；该版本现已由第 19 节的 11/5/0 协议替代
- [x] 18.3 删除 clean-inner、group-safe/time-block、旧 split mode/ratio/seed CLI 与旁路脚本，canonical/PCPF 统一绑定唯一 protocol
- [x] 18.4 分离 split seed 与 train seed，默认不构建 test，并将 protocol/manifest/seed/group counts 写入启动输出、配置快照、checkpoint 与结果 provenance
- [x] 18.5 增加确定性、不同 seed、group/weather/scene 结构、异常数据、旧协议拒绝、窗口边界、默认 test 封存和 seed 独立测试
- [x] 18.6 只生成并验证 seed 0 manifest/report，构建最小 train/validation loader，不运行其他 split seed、正式 test 或长训练
- [x] 18.7 运行 focused tests、OpenSpec strict、CLI config、compile 与 quick 验证并核对源码/本地产物边界

## 19. MMW 11/5/0 开发协议

- [x] 19.1 更新 proposal、design、current/delta specs，明确每 scene 一个 validation held-out、11 train/5 validation/0 test 与 test 入口缺失
- [x] 19.2 将 trajectory protocol、配置绑定、CLI、runner、evaluator 和分析器迁移为 train/validation-only，并删除旧 test 授权分支
- [x] 19.3 更新 focused tests、README、维护文档与固定计数/provenance，确保旧 11/2/3 manifest 和缓存失败关闭
- [x] 19.4 重新生成 seed 0 manifest/audit/report，并重建 train-only normalization、GPS 与 train/validation sparse-CSI cache/bundle
- [x] 19.5 运行 focused tests、OpenSpec strict、CLI config、compile、quick/full 回归并核对本地产物边界

## 20. MMW ID-stratified block 70/15/15 协议

- [x] 20.1 更新 proposal、design、current/delta specs，定义 `mmw_id_stratified_block_v1`、verified weather mapping、128-base-frame block、70/15/15、默认 test 封存与旧协议迁移
- [x] 20.2 基于 strict `seq_index` 和显式 frame 列表实现稳定基础样本映射、连续 block 构建、确定性标签平衡 assignment 与 block 内窗口 materialization
- [x] 20.3 实现 manifest 复用/显式 regenerate、统一 leakage validator、JSON/Markdown 报告与简单连续 block baseline 分布对比
- [x] 20.4 将 canonical/PCPF 配置、CLI、loader、日志、checkpoint/result provenance 和 split-specific cache identity 迁移到新协议，并默认不构建 test
- [x] 20.5 删除旧 trajectory-disjoint 函数、协议字符串、配置/脚本入口、fallback 与 current capability；保留历史本地产物但拒绝加载
- [x] 20.6 增加确定性、seed 独立、天气/block/base/window 泄漏、覆盖、比例、标签优化、旧 manifest/cache、train-only 统计、默认 test 封存和输入遍历顺序测试
- [x] 20.7 只生成并验证 seed 0 manifest/report，构建 train/validation loader并运行 1--2 batch 模态/CSI identity smoke；训练/验证流程不构建 test loader，不运行其他 seed 或长训练

## 21. 五个独立单模态 Stage 1 诊断

- [x] 21.1 定义 fixed-single-modality 的训练/validation 一致 mask、Stage 1-only、missing-matrix 关闭与 sealed-test 契约
- [x] 21.2 实现 fixed mask、validation 绑定和 PCPF resolver 的窄诊断参数，并增加 fail-closed focused tests
- [x] 21.3 使用当前 seed 0 ID-block manifest 完成真实 batch 64 CUDA smoke，核对五份 resolved config、GPU、输出、seed、预算和不覆盖策略
- [ ] 21.4 在独立 GPU/输出目录并行启动 image/radar/GPS/LiDAR/CSI seed1 Stage 1，监控首轮健康度并汇总 validation-best 与逐 epoch peak Top-1

## 22. ID-block 条件标签分布修复

- [x] 22.1 更新 proposal、design、current/delta specs，定义 32-base-frame block、scene/trajectory 条件标签目标、manifest v2 与旧 cache 失效
- [x] 22.2 实现确定性 conditional assignment、可扩展 quota 搜索、manifest 身份与全局/条件分布报告
- [x] 22.3 增加旧 manifest 拒绝、条件目标、默认配置、确定性与 leakage focused tests
- [x] 22.4 显式重建并审计 seed 0 manifest/report、train-only normalization、GPS 与 train/validation sparse-CSI split bundle，不启动长训练或访问 test evaluator
- [x] 22.5 运行 focused tests、OpenSpec strict、CLI config、compile、quick 与 full 验证，记录未执行的长训练

## 23. 拓扑原型监督与动态融合 2x2 消融

- [x] 23.1 定义 E0--E3、拓扑监督关闭边界、三 seed 配对、共享专家约束与交互项报告契约
- [x] 23.2 增加最小 topology-loss-off Stage 1 overlay 和配置回归测试，完成 strict OpenSpec、focused test、resolve 与 preflight
- [x] 23.3 在独立 GPU/输出目录启动 split seed 0、train seed 1/2/3 的两条专家链，并从各自 Stage 2 分叉 Static/Dynamic；禁止覆盖和自动重试，监控首轮健康度
- [ ] 23.4 完成 E0--E3 同 validation identity 评估，汇总逐 seed/均值/不确定性、E1-E0、E3-E1、E2-E0 与交互项；保持 test 未访问和 claim-ineligible
