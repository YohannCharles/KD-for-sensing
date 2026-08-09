## ADDED Requirements

### Requirement: PCPF-T forward 必须暴露无状态 beam posterior statistics

在保留既有 `fused_probability`、`unimodal_probabilities`、`logits` 和 checkpoint state dict 的前提下，PCPF-T forward MUST 追加由 `fused_probability` 确定性派生的 beam posterior statistics。新增字段 MUST 包含 MAP beam、circular mean、resultant length、circular variance、beam-space variance、beam-space spread、normalized entropy 和稳定 Top-L indices/probabilities；它们 MUST 不创建参数、buffer、loss 或 optimizer group，且旧 checkpoint MUST strict-load。

#### Scenario: 四模态 forward

- **WHEN** canonical image/radar/gps/lidar 输入经过任意合法 missing mask
- **THEN** `fused_probability` 与新增 statistics MUST 具有相同 batch 维，缺失模态不得被统计 helper 单独重新纳入

#### Scenario: 训练图隔离

- **WHEN** trainer 在 BF16/FP32 下执行 forward/backward
- **THEN** posterior statistics MUST 为 detached diagnostic tensors，loss 和梯度集合 MUST 与变更前一致

#### Scenario: 旧 checkpoint 恢复

- **WHEN** 加载已有 validation-best PCPF-T checkpoint
- **THEN** state dict key、参数数量和 strict-load 结果 MUST 不因新增 statistics 改变

## MODIFIED Requirements

### Requirement: sensing-guided probing diagnostic 必须隔离策略与 radio ground truth

系统 MUST 提供不训练模型的 validation-only finite beam probing diagnostic。主诊断 MUST 绑定预注册 topology-on PCPF Stage 3 validation-best checkpoint、与其完全匹配的 unbounded 31-mask evidence、唯一 `mmw_id_stratified_block_v1` validation identity、正式 ULA-DFT topology，以及只从同一 protocol train role 拟合的 topology likelihood artifact。诊断 MUST 评估 CSI availability 严格为 false、image/radar/gps/lidar 任意非空组合的全部 15 个 mask；结果 MUST 保持 `claim_ineligible=true`、`outer_test_accessed=false`，不得构建 test loader、训练/更新模型或根据 validation 切换 checkpoint。

TBCP-7 candidate selection MUST 只依赖冻结 sensing posterior、train-only topology likelihood、已请求 beam indices 与其返回 measurement；静态/一轮反馈 baseline MUST 遵守相同 RF query 边界。完整 validation 64-beam power、GT、channel、未请求 CSI/gain 或 metric denominator MUST NOT 进入任一非 oracle candidate policy。radio simulator MAY 私有缓存完整 validation power vector，但 public `probe` 接口 MUST 只返回显式请求的 measurement，最终 beam MUST 只从 K=7 返回值中选择。

#### Scenario: 运行 15-mask TBCP-7 diagnostic

- **WHEN** evaluator 对 topology-on seed 1/2/3 运行固定 K=7 TBCP-7 与 matched baselines
- **THEN** 每个 mask MUST 使用相同完整 validation sample identity/order、同一个 train-only likelihood、相同 simulator/final selection/metric
- **AND** 报告 MUST 输出逐 mask、Full、drop-1、drop-2、Single macro/worst 和全部 seed mean/std，不得选择最优 seed

#### Scenario: candidate policy 尝试读取 oracle 信息

- **WHEN** 任一非 oracle policy 收到 GT、channel、未请求 CSI、完整 beam-power vector或 metric denominator，或 simulator 返回未请求 beam
- **THEN** API/断言 MUST 阻止该路径，diagnostic 不得生成可用报告

#### Scenario: validation label 与 radio ground truth 漂移

- **WHEN** Full-64 argmax 不等于 GT，或无噪声 probe 策略的 correct 与 GT coverage 不一致
- **THEN** diagnostic MUST 失败并报告 label/power/tie 漂移，不得静默汇总
