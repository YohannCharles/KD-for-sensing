# training-evaluation-runtime Specification

## Purpose

定义 MMW T2/baseline 与受限 DeepSense6G T2 的共享训练与评估边界，保证 T2 的 same-model consistency 只在训练期执行，并保持可审计 provenance。
## Requirements
### Requirement: T2 runtime 仅保留 same-model consistency

训练 runtime MUST 仅保留 T2 所需的 embedded full-modal teacher CE、same-model temporal superset consistency，以及 active PCER direction search 的同模型 full-to-masked、on-policy cached-evidence target 和 balanced LOMO consistency。Evaluation MUST 不执行训练 target forward；T2 disabled path 与 S1 MUST 不产生额外 forward 或外部 artifact 读取。

#### Scenario: S1 关闭 superset consistency

- **WHEN** S1 recipe 关闭 superset consistency 且不属于 direction search
- **THEN** trainer MUST 不保存 superset payload 或执行第二次 model forward
- **AND** 训练仍能计算共享 beam、embedded teacher CE、BPA 和 router loss

#### Scenario: Direction search target

- **WHEN** B2/B3/B4 计算 supervised route target
- **THEN** target MUST detach 且 predicted router MUST 保持梯度
- **AND** on-policy removal MUST 只重跑缓存 evidence 上的轻量 router/fusion，不得重复 backbone

### Requirement: 评估使用 recipe 声明的数据集 protocol

evaluation MUST 复用共享四模态 batch/input contract，并以 recipe 声明的 MMW 或 DeepSense6G 数据集、checkpoint、split 和 mask identity 产出指标。PCER direction search MUST 使用与历史 quick PCER 相同的 development split 和 deterministic S0-S5 mask，并保留每个 S3 缺失模态。DeepSense6G evaluation MUST 不执行 retired branch 或外部 teacher path。

#### Scenario: 评估 current checkpoint

- **WHEN** 用户评估任一 current recipe 或 active direction-search checkpoint
- **THEN** runtime MUST 不执行 retired branch 或外部 teacher path
- **AND** 输出 MUST 带有足以比较的 recipe、dataset、scene 或 domain、seed、split、mask 与 checkpoint provenance

### Requirement: 训练与评估 runtime 必须保持证据和资源边界
训练、package evaluation 和固定 mask evaluation MUST 对相同 checkpoint 应用一致的 CUDA runtime、profile、GPS 和 normalization 校验。evaluation owner MUST 在退出时关闭其创建的 dataloader workers；默认 evaluation MUST 流式累计指标，除非调用方显式请求 prediction capture。

#### Scenario: 固定 mask 评估 current checkpoint
- **WHEN** fixed-mask evaluator 加载一个 current MMW checkpoint
- **THEN** 它 MUST 校验 checkpoint profile、GPS mode 和 normalization artifact
- **AND** 它 MUST 使用 checkpoint 保存的 train-fitted normalization artifact 而不是重新拟合 scaler

#### Scenario: 默认评估不缓存全量输出
- **WHEN** 常规 validator 或 package evaluator 运行且未请求 capture
- **THEN** runtime MUST 只保留完成 metrics 所需的聚合状态
- **AND** evaluation 完成或失败后 MUST 关闭创建的 worker

### Requirement: development 与 partial 运行必须明确隔离
development 或 partial evaluation MUST 记录其实际 sample/domain/mask 覆盖范围，并不得伪装为完整 comparison evidence。

#### Scenario: 用户限制 batch 或 domain
- **WHEN** evaluator 使用 `max_batches` 或 `max_domains`
- **THEN** 输出 MUST 标记 `development_partial=true` 并记录实际计数
- **AND** 正式 summary MUST 拒绝该输出

### Requirement: 训练初始化 checkpoint 必须与轨迹续跑分离
runtime MUST 提供仅加载模型权重并重置训练状态的显式 initialization checkpoint契约，并 MUST 与严格 `training.resume` 互斥。初始化 MUST 验证source SHA、checkpoint role/schema、加载key及新增missing key allowlist。

#### Scenario: 从成熟 Current checkpoint 初始化候选
- **WHEN** 候选配置声明 initialization checkpoint
- **THEN** runtime MUST 在optimizer构建前加载允许的expert与Current Router权重
- **AND** optimizer、scheduler、epoch、RNG、sampler和extension state MUST 从新run重新开始
- **AND** load report与source SHA MUST 写入checkpoint provenance

#### Scenario: 初始化身份不一致
- **WHEN** source SHA、shape、既有required key或unexpected key不符合声明
- **THEN** runtime MUST 在训练启动前失败

### Requirement: Router 校准必须冻结并固定 expert 运行状态
候选校准 MUST 将encoder、projection、reliability head、active/inactive beam head、temporal pooling和Current Router参数设为不可训练，并 MUST 在model进入train模式后继续令其BN/Dropout保持eval。optimizer MUST 只包含声明的候选 Router参数。

#### Scenario: 执行校准 optimizer step
- **WHEN** 一个候选batch完成backward和step
- **THEN** 只有candidate Router参数 MAY 变化
- **AND** frozen expert的参数与running statistics MUST 保持不变

### Requirement: 配对联合退化运行时必须传播固定 provenance
runtime MUST 使用内容寻址的240-entry Joint training panel，并 MUST 在run、checkpoint和summary中传播panel checksum、监督类型、source checkpoint和inner-only claim状态。

#### Scenario: 重复生成相同筛选任务
- **WHEN** seed、panel版本、source checkpoint和候选配置相同
- **THEN** resolved config与panel checksum MUST 相同
- **AND** train/evaluation corruption随机流 MUST 与其他seed或角色隔离

### Requirement: 物理效用筛选身份冻结
夜间动态 Router 决策对齐筛选 MUST 在启动前冻结 source checkpoint、source SHA、Joint panel SHA、loss source SHA、候选架构、决策目标、seed、batch、epoch 和 GPU 映射；已有 manifest 与请求不一致时 MUST fail closed。

#### Scenario: 八卡固定矩阵
- **WHEN** 启动完整 seed1 筛选
- **THEN** 系统在 GPU0--7 一卡一任务运行 `PATR/H2R × 四决策目标`，并为每项保存 resolved config、日志和状态

#### Scenario: 禁止身份漂移续跑
- **WHEN** 输出目录已有 manifest 且当前请求的任一冻结字段不同
- **THEN** launcher 拒绝复用该输出目录

### Requirement: Inner-only 晋级边界
决策对齐筛选结果 MUST 保持 inner-only 且 `claim_eligible=false`；只有使用冻结 Joint evaluator 通过既有材料性、置信区间与非劣 Gate 后才可规划 seed2--5。

#### Scenario: 夜间训练不自动形成正式 claim
- **WHEN** 八个 seed1 训练任务完成
- **THEN** 系统保留 checkpoint 和训练证据，但不得自动修改 canonical recipe、正式 claim 或 Gate

### Requirement: PCER quick-validation 使用验证集最佳 checkpoint
训练 runtime MUST 以 opt-in 配置逐 epoch运行 validation，并在 validation loss 改善时发布独立 `best.pth`。默认 fixed-epoch `last.pth` 行为 MUST 不变，测试集 MUST 不参与 checkpoint 选择。

#### Scenario: 选择 quick-validation checkpoint
- **WHEN** PCER 训练在多个 epoch 产生 validation loss
- **THEN** `best.pth` MUST 对应最低有限 validation loss并记录 epoch
- **AND** 最终固定 mask 评测 MUST 加载该 checkpoint

### Requirement: PCER 固定评测按样本身份生成 mask
evaluation runtime MUST 以 global eval seed、stable sample identity、mask type 和 variant 生成逐样本固定 mask，并在 A0--A3 间校验 identity 一致。S1 MUST 使用三个 variant；S3 MUST 分别覆盖每个模态；S5 MUST 在测试集上均衡合法模态对和 recent burst template。

#### Scenario: 固定六场景评测
- **WHEN** evaluator 运行 S0--S5
- **THEN** MUST 输出 Top-1、Top-3、Top-5、Within-3、beam-index MAE 和已有通信指标
- **AND** S3 MUST 输出每模态、macro 和 worst，S5 MUST 输出 macro 和 worst pair

### Requirement: quick-validation 证据保持开发边界
四组 PCER quick-validation MUST 标记为单 seed、inner/development、claim-ineligible。系统 MUST 不把本轮结果写入正式 claim，也 MUST 不自动运行多 seed 或剩余实验矩阵。

#### Scenario: 汇总完成
- **WHEN** comparison report 已生成
- **THEN** runtime MUST 停止在四组结果和预注册判断
- **AND** 下一批实验只能作为建议，不得自动启动
