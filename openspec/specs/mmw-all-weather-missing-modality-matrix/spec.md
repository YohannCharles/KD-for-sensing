# mmw-all-weather-missing-modality-matrix Specification

## Purpose
定义 MMW 15-domain 全天候缺失模态主矩阵的数据盘点、等域采样、公平训练、固定缺失评估与表征诊断契约。
## Requirements
### Requirement: MMW all-weather domain inventory
系统 MUST 将 sunny、rainy、foggy 下 5 个已声明 Town03 scenario 解析为 15 个显式 domain，并在训练前验证每个 domain 的 manifest、split metadata、sensing modality artifacts 和样本数。缺少任一 domain 或 strict eligibility 不通过时 MUST fail closed。

#### Scenario: 15 个 domain preflight 通过
- **WHEN** 用户启动 MMW all-weather matrix
- **THEN** preflight MUST 输出恰好 15 个唯一 `condition/scenario` domain
- **AND** 每个 domain MUST 具有非空 train 与 validation CSV、strict split metadata、image/GPS/LiDAR/radar 输入和 beam target
- **AND** preflight MUST 记录 excluded sensitive fields 为 CSI、channel、mmWave、beam-power、path 和 radio labels

#### Scenario: domain artifact 不完整
- **WHEN** 任一 domain 缺少 radar map、split metadata、CSV 或 strict eligibility
- **THEN** launcher MUST 不启动任何正式训练
- **AND** 报告 MUST 给出 domain、缺失 artifact 和公开 preparation 修复命令

### Requirement: Domain-balanced pooled training
系统 MUST 为 all-weather pooled train dataset 按 domain 等权采样，而不是按原始 domain 样本数加权。sampler seed、每 domain 样本数、每样本权重和每 epoch 抽样数 MUST 写入运行 provenance。

#### Scenario: 不同大小 domain 等权
- **WHEN** 两个 domain 分别包含 100 和 400 个 train samples
- **THEN** 每个 domain 在抽样分布中的总权重 MUST 相等
- **AND** 单样本权重 MUST 分别与 `1/100` 和 `1/400` 成比例

### Requirement: Four-method fair screening matrix
系统 MUST 在相同 15-domain 数据、四 sensing modalities、missing augmentation、固定 epoch、optimizer budget 和 checkpoint policy 下运行 S1、T2、AMBER-Full 与 RMBP-MM seed1。screening MUST 使用 last fixed-epoch checkpoint，不能用 validation 指标选择 best checkpoint。

#### Scenario: GPU0-3 四方法并行
- **WHEN** preflight 与四方法 smoke 全部通过
- **THEN** launcher MUST 在 GPU0、1、2、3 各启动一个方法 seed1
- **AND** 每卡 MUST 至多一个训练进程
- **AND** GPU4-7 上现有 DeepSense candidate 进程 MUST 不被 launcher 修改

### Requirement: All-weather missing-modality evaluation
评估 MUST 对每个方法使用相同样本、相同 whole-modality pattern 和相同 temporal mask identity，并输出 per-domain、per-weather、per-scene、domain macro 和 worst-domain 指标。距离敏感指标 MUST 使用经 MMW codebook provenance 确认的 geometry。

#### Scenario: 四模态完整缺失组合
- **WHEN** enabled sensing modalities 为 image、radar、GPS、LiDAR
- **THEN** evaluation MUST 覆盖全部 15 个非空 available-modality subsets
- **AND** summary MUST 分别报告 clean、Drop1、Drop2、Drop3 和 single-modality 结果

#### Scenario: 固定 temporal missing matrix
- **WHEN** evaluation 运行 H5/P1 temporal missing
- **THEN** 所有方法 MUST 共享 0、20、40、60、80% rates 和固定 mask cache
- **AND** 0% MUST 只评估一个 clean mask，非零 rate 的 `frame_level` 与 `block` MUST 覆盖全部互异几何，`modality_frame` MUST 使用多个互异固定 mask
- **AND** evaluator MUST fail closed 拒绝 mask 数、type、shape、rate 或互异几何覆盖与请求不一致的旧 cache
- **AND** summary MUST 先计算每个 mask 的 15-domain macro，再报告 type-equal mean、mask 间标准差、最差值、实际缺失率和末帧可用性
- **AND** paired summary MUST 校验 domain、sample、mask identity 与 cache provenance 后才能计算 delta

#### Scenario: baseline 论文范围与本地适配分离
- **WHEN** summary 报告 AMBER-Full 或 RMBP-MM
- **THEN** provenance MUST 记录 paper-equivalent 与 local-adaptation scope
- **AND** AMBER-Full MUST 说明本地协议未使用原文 historical beam 且模型/损失为缩小近似实现
- **AND** RMBP-MM MUST 说明本地协议未实现 partial beam、单模态预训练和 label-guided similarity imputation
- **AND** RMBP-MM temporal missing MUST 标记为 out-of-paper-scope diagnostic，不能作为论文等价曲线

#### Scenario: 85/90/95% 极端 modality-time cell 缺失
- **WHEN** evaluation 在 5 帧 × 4 模态输入上运行 85%、90%、95% temporal missing
- **THEN** mask MUST 分别精确丢失 17、18、19 个 modality-time cells
- **AND** 每个 mask MUST 分别只保留 3、2、1 个 cells
- **AND** 该扩展 MUST 只使用 `modality_frame`，不得用重复的 80% frame/block 几何冒充 85% 或 90%
- **AND** summary MUST 将极端稀疏曲线与原三类型 type-equal 主曲线分开报告

### Requirement: Weather improvement gate
第一轮 MUST 直接评估 T2 与现有 reliability fusion，不得引入 weather label 输入或天气专用模型模块。后继改进只有在 weather/domain paired diagnostics 展示可复现失败模式时才能启动。

#### Scenario: reliability fusion 已适应天气
- **WHEN** T2 在 rainy/foggy 的 macro missing Top1、worst-domain 与 calibration 均不劣于 S1 门禁
- **THEN** 主线 MUST 保持原 T2，不新增天气模块

#### Scenario: 发现天气相关过度自信
- **WHEN** 至少两个独立 weather/domain 对显示同一 modality gate 过度自信且伴随显著负迁移
- **THEN** summary MUST 记录失败模式
- **AND** 新 reliability calibration 或 quality-aware module MUST 通过独立 OpenSpec change 提出

### Requirement: Paired fused-representation stability diagnostic

系统 MUST 支持对 T2、AMBER-Full 与 RMBP-MM 的固定 epoch checkpoint 提取实际 beam head 输入融合表征，并在相同 validation sample、相同 temporal mask identity 下比较 clean 与 20%、40%、60%、80% `modality_frame` 缺失。诊断产物 MUST 留在 ignored `outputs/`，不得修改模型 forward、训练 loss 或 checkpoint。

#### Scenario: 方法内 clean PCA 投影

- **WHEN** 系统生成三方法 clean 与 missing 融合表征图
- **THEN** T2 MUST 使用二维 fused `output_features`，modular baseline MUST 使用最终预测时刻 fused `output_features[:, -1, :]`
- **AND** 每种方法 MUST 只使用自己的 L2-normalized clean 表征拟合 PCA，再用同一基底投影该方法全部 missing 表征
- **AND** 三个独立模型的原始潜在坐标 MUST NOT 直接叠加后把绝对位置解释为方法差异

#### Scenario: 缺失表征配对漂移统计

- **WHEN** 诊断消费共享 v2 mask cache
- **THEN** 20%、40%、60%、80% MUST 使用全部固定 `modality_frame` mask，并校验 sample、domain、rate、mask digest 和 checkpoint provenance
- **AND** 原始归一化特征空间 MUST 报告 paired cosine distance、最近 clean beam centroid 的圆周 beam 距离、clean-to-missing centroid assignment 保持率和预测 beam 圆周偏移
- **AND** summary MUST 按 rate 报告 mask mean、standard deviation、worst mask，并输出 per-domain 分层结果
- **AND** PCA 图 MUST 只作辅助展示，不得用二维投影距离替代上述高维指标

#### Scenario: Alignment Loss 结论边界

- **WHEN** T2 的表征漂移优于 AMBER-Full 与 RMBP-MM
- **THEN** 报告 MAY 将结果描述为 prototype-aligned T2 与缺失表征稳定性相关
- **AND** 报告 MUST 说明三架构比较不能隔离 Beam Prototype Alignment Loss 的因果贡献
- **AND** 正式 loss-effectiveness claim MUST 要求同一 T2 架构关闭 alignment loss 的 matched ablation

#### Scenario: 原始高维循环拓扑可视化

- **WHEN** 诊断展示 learned T2 prototypes 或三方法 clean class centroids 的循环邻接
- **THEN** 系统 MUST 输出原始 L2-normalized 64 维 cosine Gram matrix 或等价原空间相似度证据
- **AND** 系统 MUST 输出 similarity 随 circular beam distance 的统计以及原空间 Topo@1/Topo@3
- **AND** prototype 无监督二维流形投影 MUST 只使用原始特征距离构图；class centroid MAY 由真值标签聚合，但投影 MUST 只使用 centroid distance，且图注 MUST 披露 label-conditioned centroid
- **AND** 三方法 centroid 投影 MUST 使用相同邻居数和算法，不得按方法挑选最优参数
- **AND** 报告 MUST 明确 PCA、Isomap 或谱嵌入只作展示，不替代原空间指标

#### Scenario: 缺失后的 signed feature shift

- **WHEN** 诊断展示 20%、40%、60%、80% 缺失对融合表征的影响
- **THEN** 每种方法 MUST 使用自己的 leave-one-out clean class centroid bank 在原始归一化空间计算 clean 与 missing assignment
- **AND** signed circular offset MUST 使用稳定的 `-32..31` 口径并保留 `63/0` wrap-around
- **AND** 三方法 MUST 使用相同 validation samples、全部固定 `modality_frame` masks 和共享色标
- **AND** shift histogram MUST 先在每个 domain 内归一化，再对 15 domain 等权平均
- **AND** 图与 summary MUST 区分 feature-assignment shift 和 prediction shift，不得用平均箭头掩盖样本级双向偏移
