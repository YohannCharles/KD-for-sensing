## MODIFIED Requirements

### Requirement: T2/baseline 与双数据集是唯一 current 研究 surface
系统 MUST 将 MMW 的 T2、S1、MaskTrain-CLS、AMBER-Full、RMBP-MM、AMR-Net-4M-Adapted，以及 DeepSense6G Scene31--34 的 T2、MaskTrain-CLS、AMBER-Full、RMBP-MM、AMR-Net-4M-Adapted 四模态路径和其证据依赖视为 current source surface。AMR MUST 作为 `modular_sequence` representation core 实现，不得恢复 retired `amr_net` whole-model 或旧 runner。

#### Scenario: 构建新增 baseline
- **WHEN** config 构建 MaskTrain-CLS 或 AMR-Net-4M-Adapted
- **THEN** model type MUST 仍为 `modular_sequence`
- **AND** 四模态顺序、外部 missing mask 和 shared runtime contract MUST 与其他方法一致
- **AND** metadata MUST 分别标记 simple mask-training control 或 local AMR adaptation

### Requirement: DeepSense6G 次级证据必须独立且可重现
系统 MUST 以 `deepsense6g_twc_secondary_v1` 将 Scene31--34 合并为一个完整数据集，并绑定五方法、seed `(1,2,3)`、固定训练 schedule 和固定评估 mask。每个 method/seed MUST 仅训练一个 pooled checkpoint，总训练任务 MUST 为 15；协议不得把四个 scene 拆成独立训练任务。协议 MUST 从只读原 CSV 生成内容寻址派生 CSV，只保留 `future_beam1` 恰有 64 个有限非负功率值的样本，并记录 source/derived hash、计数与 rejected-row hash；`num_pred=1` MUST 不加载额外 future horizon。

#### Scenario: 构建 pooled DeepSense6G 训练单元
- **WHEN** launcher 构建一个 method/seed config
- **THEN** config MUST 同时引用 Scene31--34 的四个 train/test 派生 CSV
- **AND** GPS normalization MUST 只在合并训练集上统一拟合
- **AND** run/checkpoint identity MUST 不包含单独 scene 训练维度

#### Scenario: 原始 future beam 文件含 NaN
- **WHEN** DeepSense6G source CSV 中某行的 `future_beam1` 不满足 64 个有限非负值
- **THEN** protocol preparer MUST 从派生 CSV 剔除该行并记录审计计数/hash
- **AND** 系统 MUST 不插值 beam power 或伪造 target label

#### Scenario: 汇总 DeepSense6G 次级证据
- **WHEN** 五方法三 seeds 的 pooled fixed-mask 评估全部完成
- **THEN** summary MUST 聚合 15 个 pooled checkpoint 的跨 seed 结果
- **AND** MMW split、mask、normalizer 和 summary row MUST 不得混入

### Requirement: 双数据集评估必须输出通信、机制与复杂度证据，并支持可选 corruption 扩展
MMW 与 DeepSense6G fixed evaluator MUST 从完整 future beam-power vector 计算 normalized gain、gain loss dB，以及参考 oracle SNR 0/10/20 dB 的 spectral-efficiency ratio/rate loss。系统 MUST 生成 clean/Block80 的 physical error CDF、far-error、feature drift、prototype-neighbor margin 和 router-oracle alignment；推理期 corruption MUST 使用固定 spec/seed 且不修改 checkpoint/training recipe，但 MUST 默认保持关闭，不得进入默认 post-hoc 队列；只有显式 `--run-reliability-stress` 才可生成和运行 corruption shards；复杂度 MUST 记录 params、可用 MACs、batch1/64 latency、throughput、peak memory、硬件和 AMP policy。

#### Scenario: 运行推理期 corruption
- **WHEN** evaluator 应用 GPS、image、Radar 或 LiDAR corruption
- **THEN** corruption spec、severity、seed 和 parent protocol identity MUST 写入版本化 manifest
- **AND** training recipe 与 checkpoint identity MUST 保持不变

#### Scenario: 默认不运行可靠性压力测试
- **WHEN** launcher 未收到 `--run-reliability-stress`
- **THEN** post-hoc manifest MUST 不包含 corruption jobs，且 reliability stress MUST 标记为 disabled
- **AND** 默认队列不得生成 corruption summary 或可靠性 claim

#### Scenario: 显式运行可靠性压力测试
- **WHEN** 用户明确传入 `--run-reliability-stress`
- **THEN** launcher MUST 生成包含固定 corruption spec/seed 的 manifest，并将 opt-in 状态写入 request/payload
- **AND** 只有该 manifest 可调度 corruption jobs
