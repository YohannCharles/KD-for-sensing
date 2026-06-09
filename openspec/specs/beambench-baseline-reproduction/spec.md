# beambench-baseline-reproduction Specification

## Purpose
定义 BeamBench/DeepSense6G baseline 复现实验的官方源码审计、数据检查、mock smoke、Image AE + GPS Direct 本地训练与四场景报告交付契约，确保复现结果、指标口径、产物边界和后续 attention 插入点可审计。
## Requirements
### Requirement: 官方 BeamBench 源码与环境审计
系统 MUST 提供 BeamBench 官方 baseline 复现前置审计，记录官方仓库来源、commit hash、目录结构、训练入口、评估入口、模型保存路径、数据路径、配置文件位置、依赖版本要求和当前环境偏差。审计结果 MUST 写入 `ENVIRONMENT.md`、`DATASET_STRUCTURE.md`、`BASELINE_REPORT.md` 或 `results/reproduce_baseline.md` 中的对应章节。

#### Scenario: 记录官方仓库信息
- **WHEN** 开发者执行 BeamBench 源码审计
- **THEN** 系统 MUST 记录官方仓库 URL、clone 或 checkout 的 commit hash、README 推荐评估命令、`challenge.py` 或等价评估入口、配置目录和默认 `results/models`、`results/topk` 路径

#### Scenario: 记录官方环境要求
- **WHEN** 审计读取官方 README 或 Dockerfile
- **THEN** 系统 MUST 记录官方声明的 Ubuntu、CUDA、Python、PyTorch 和关键 pip 依赖要求
- **AND** 系统 MUST 记录当前 `kd_mm_beam` 环境的 Python、CUDA、PyTorch、torchvision、GPU 型号和可用性

#### Scenario: 官方源码或权重不完整
- **WHEN** 官方仓库缺少被入口引用的源码文件、模型权重或数据文件
- **THEN** 系统 MUST 在报告中明确列出缺失项、受影响的 baseline 和下一步解决方案
- **AND** 系统 MUST NOT 把无法运行的官方 baseline 标记为已复现

### Requirement: BeamBench 数据接口检查
系统 MUST 提供 `scripts/check_dataset.py` 或等价薄入口，用于检查 BeamBench/DeepSense6G 数据目录和 CSV 字段。该检查 MUST 不修改、不移动、不删除真实数据文件。

#### Scenario: CSV 存在性与字段检查
- **WHEN** 用户运行 `conda run -n kd_mm_beam python scripts/check_dataset.py --data-root <path> --csv <file>`
- **THEN** 系统 MUST 验证 CSV 文件存在且可读取
- **AND** 系统 MUST 报告 camera、LiDAR、radar、GPS、label、scene id、sample id 和 sequence id 相关字段是否存在或可由配置解析

#### Scenario: 传感器文件引用检查
- **WHEN** CSV 行包含 camera、LiDAR、radar 或 GPS 文件路径列
- **THEN** 系统 MUST 按数据根目录解析每一行引用的文件路径
- **AND** 系统 MUST 统计每个模态的存在数量、缺失数量、缺失比例和示例缺失路径

#### Scenario: label 和 beam index 检查
- **WHEN** CSV 或 beam label 文件提供 optimal beam index、future beam label 或等价目标
- **THEN** 系统 MUST 验证 label 存在且位于配置声明的合法范围内
- **AND** 系统 MUST 报告 label 总数、非法 label 数量、label 最小值、最大值和是否使用 0-based 或 1-based beam shift

#### Scenario: scene sample sequence 标识解析
- **WHEN** CSV 包含 scene、sample、timestamp、frame 或 sequence 字段
- **THEN** 系统 MUST 解析并报告 scene ID、sample ID、sequence ID 或 timestamp 的可用性
- **AND** 系统 MUST 在不能解析时报告缺失字段名和可选 fallback，而不是静默通过

### Requirement: mock dataset smoke 流程
当真实 DeepSense6G/BeamBench 数据或官方 checkpoint 暂时不可用时，系统 MUST 提供极小 mock dataset 以验证完整代码路径。mock dataset MUST 只用于 smoke test，所有输出、日志和报告 MUST 显式标注 `MOCK`。

#### Scenario: 真实数据不可用时运行 mock
- **WHEN** 用户选择 mock 模式或真实数据检查失败后显式运行 mock smoke
- **THEN** 系统 MUST 创建或读取极小 mock CSV 和必要传感器占位输入
- **AND** 系统 MUST 完成 dataloader、model forward、loss、metric、checkpoint save/load 和 evaluation

#### Scenario: mock 结果不能冒充真实结果
- **WHEN** mock pipeline 生成 metrics、checkpoint、日志或报告
- **THEN** 每个相关 artifact MUST 包含 `mock_data: true` 或 `MOCK` 标记
- **AND** `BASELINE_REPORT.md` 和 `results/reproduce_baseline.md` MUST 明确说明 mock 指标不得用于论文或官方结果比较

#### Scenario: mock 保持关键模态和指标路径
- **WHEN** mock pipeline 覆盖一个多模态 baseline
- **THEN** 系统 MUST 不为了跑通而删除该 baseline 声明启用的关键模态
- **AND** 系统 MUST 仍计算 DBA 和 top-k accuracy 或记录该 metric 无法计算的具体原因

### Requirement: 官方 baseline 训练与评估 wrapper
系统 MUST 提供 BeamBench baseline 的训练和评估入口，优先保留官方 `challenge.py` 评估语义；如官方已有训练入口则通过 wrapper 调用，如官方训练入口不完整则使用本仓库现有 `kd_sensing` 架构实现可测试 baseline，并在报告中区分官方评估、等价实现和 mock smoke。

#### Scenario: 官方推荐评估命令复现
- **WHEN** 用户具备官方数据、官方权重和兼容环境并运行评估 wrapper
- **THEN** 系统 MUST 调用官方 `challenge.py` 或语义等价入口
- **AND** 系统 MUST 记录实际 command、工作目录、官方 commit、data folder、CSV、type list、seed、checkpoint 路径和预测 CSV 路径

#### Scenario: baseline 完整训练或评估闭环
- **WHEN** 用户运行可用的 baseline 训练或评估命令
- **THEN** 系统 MUST 完成 data loading、model forward、loss computation、metric computation、checkpoint save/load 和 validation/test evaluation 中至少一个完整闭环
- **AND** 系统 MUST 保存或记录 checkpoint 路径、metrics、日志路径和是否使用 mock data

#### Scenario: baseline 模态列表清晰
- **WHEN** 系统列出 BeamBench 可复现 baseline
- **THEN** 报告 MUST 至少覆盖 Camera AE + GPS、late fusion of pretrained modality features、GPS dense、camera/radar/LiDAR/radar/GPS 组合或官方仓库中可识别的多模态配置
- **AND** 不可运行的 baseline MUST 标记为 blocked、missing source、missing checkpoint、missing data 或 unsupported environment

### Requirement: Arnold22 Image AE + GPS Direct 本地训练
系统 MUST 为 Arnold22 BeamBench Table III 中 `Camera=AE, GPS=Direct, Fusion=Yes` 目标行提供项目内本地训练实现。该实现 MUST 不以 residual/gated/attention 模型替代论文行，且 MUST 在报告中区分本地训练数值与官方 Table III 数值。

#### Scenario: 论文目标行模型结构
- **WHEN** 用户运行 Image AE + GPS Direct 本地训练入口
- **THEN** 模型 MUST 使用 Camera AE encoder 产生 image latent
- **AND** 模型 MUST 使用 GPS direct feature encoder
- **AND** 模型 MUST 将 image latent 与 GPS direct feature 融合后输出 64-beam classifier logits

#### Scenario: 本地训练闭环
- **WHEN** 本地 DeepSense6G scene31-34 sequence CSV 可用
- **THEN** 系统 MUST 从 CSV 读取 camera、GPS/BS GPS 和 future beam label
- **AND** 系统 MUST 支持先训练或加载 Camera AE checkpoint
- **AND** 系统 MUST 冻结或按配置控制 AE encoder 后训练 fusion classifier
- **AND** 系统 MUST 输出 checkpoint、predictions、history 和 BeamBench DBA/top-k metrics 到 ignored 的本地产物路径

#### Scenario: 本地结果不冒充官方结果
- **WHEN** 本地 Image AE + GPS Direct 训练或 dry-run 完成
- **THEN** 报告 MUST 说明是否使用官方 pretrained 权重和官方完整训练搜索流程
- **AND** 若未使用官方权重和官方完整流程，报告 MUST NOT 声称本地数值等同 Table III

### Requirement: Image AE + GPS Direct 训练吞吐优化
系统 MUST 为 Image AE + GPS Direct 本地训练入口提供适合 RTX 3090 和多核 CPU 的吞吐优化。优化 MUST 保持样本数、image size、训练 epoch 上限、early stopping 和 DBA 选 best 语义，不得通过减少数据、减少 epoch 或降低输入分辨率冒充加速。

#### Scenario: 冻结 AE latent cache
- **WHEN** Camera AE encoder 已冻结且用户启用 feature cache
- **THEN** 系统 MUST 将训练集和测试集 camera AE latent 预计算到当前 run 的 ignored 输出目录
- **AND** fusion classifier 训练和评估 MUST 复用这些 latent、GPS 和 label
- **AND** 该 cache MUST 可关闭，以支持非冻结 AE 或需要完全在线 forward 的实验

#### Scenario: GPU 与 DataLoader 加速配置
- **WHEN** 用户在 CUDA 设备上运行专用训练入口
- **THEN** 系统 SHOULD 支持 AMP、GradScaler、TF32、fused AdamW 和 cuDNN benchmark
- **AND** DataLoader SHOULD 支持 pin memory、persistent workers、prefetch factor 和 non-blocking transfer
- **AND** dry-run MUST 仍强制使用小样本、一轮训练和零 worker，避免 smoke test 变慢或不稳定

#### Scenario: 加速配置可审计
- **WHEN** 本地 Image AE + GPS Direct 训练完成
- **THEN** run report MUST 记录 AMP、TF32、feature cache、loader 和 batch size 等吞吐相关设置
- **AND** 文档 MUST 给出默认加速命令和关闭 cache/AMP 的调试方式

### Requirement: Camera AE + GPS Direct 四场景论文复现实验
系统 MUST 提供 Arnold22 BeamBench Table III `Camera=AE, GPS=Direct, Fusion=Yes` 行的四场景本地复现实验入口，并输出 Table III 风格的指标汇总。该入口 MUST 明确区分本地 sequence split 结果与官方完全 unseen test dataset 结果。

#### Scenario: 论文 split batch runner
- **WHEN** 用户运行四场景 Camera AE + GPS Direct 复现入口
- **THEN** 系统 MUST 支持一次性在 scenes 32、33、34 上联合训练
- **AND** 系统 MUST 分别在 scenes 31、32、33、34 上评估同一个 best checkpoint
- **AND** 每个 eval scene MUST 使用互不覆盖的输出目录
- **AND** 系统 MUST 生成全局 run report、per-scene metrics、predictions、feature cache 和 Table III 风格汇总到 ignored 本地产物路径

#### Scenario: Table III 风格汇总
- **WHEN** 四场景复现实验完成
- **THEN** 系统 MUST 输出 CSV、Markdown 和 JSON 汇总
- **AND** 汇总 MUST 包含 scene31-34、本地 overall、论文目标值和差距
- **AND** 汇总 MUST 标明 metric 字段使用 `official_top3_dba`，并说明是否使用官方权重、官方测试集、官方训练搜索流程

#### Scenario: 已训练 checkpoint 四场景 eval-only
- **WHEN** 用户提供已训练的 Image AE + GPS Direct paper-split fusion checkpoint
- **THEN** 系统 MUST 支持不重新训练 fusion，直接评估 scenes 31-34
- **AND** 系统 MUST 从 checkpoint 恢复模型配置、AE checkpoint 路径和 GPS scaler
- **AND** 系统 MUST 输出 per-scene metrics、predictions 和 Table III 风格 CSV/Markdown/JSON 汇总
- **AND** 报告 MUST 标明 eval-only 使用的是已有 checkpoint，避免和重新训练结果混淆

#### Scenario: best checkpoint 选择口径可审计
- **WHEN** 用户运行单场景或四场景训练
- **THEN** 系统 MUST 在 run report 中记录 best checkpoint 选择使用 `test_as_validation` 还是从训练集切分出的 `validation`
- **AND** 若使用 test CSV 逐 epoch 选择 best checkpoint，报告 MUST 明确标注该结果不等同官方完全 unseen test evaluation

#### Scenario: scene31 泛化专项复现
- **WHEN** 用户要求优先提升 scene31 泛化
- **THEN** 系统 MUST 支持只评估 scene31 的 paper split run
- **AND** `paper_distance_angle` MUST 使用官方 `challenge.py` 的 `arctan(x/y)` 角度公式
- **AND** scene32 的 paper 默认校准角 MUST 使用 `-0.8125375604986421 + pi/2`
- **AND** frozen AE feature cache signature MUST 包含 GPS 特征版本和 scene 校准角，避免旧 GPS cache 被复用
- **AND** 报告 MUST 区分 scene31 单项结果与完整 scenes 31-34 overall 结果

### Requirement: BeamBench 指标核对与测试
系统 MUST 核对 BeamBench DBA、top-k accuracy 和 top-3 DBA 或官方等价指标实现。若官方指标实现不完整或口径不适合当前 64-beam circular label 语义，系统 MUST 提供独立 metrics helper 并用测试说明口径。

#### Scenario: perfect prediction 指标最高
- **WHEN** predictions 的 top-1 beam 与 ground truth 完全一致
- **THEN** top-1 accuracy、top-k accuracy 和 DBA 类指标 MUST 达到最高值

#### Scenario: beam index 偏差越大 DBA 下降
- **WHEN** 两组预测相对同一 ground truth 的 beam distance 分别为小偏差和大偏差
- **THEN** 大偏差预测的 DBA MUST 小于或等于小偏差预测的 DBA

#### Scenario: top-k 命中 ground truth
- **WHEN** ground truth 出现在预测 top-k 集合中但不在 top-1
- **THEN** top-k accuracy MUST 计为命中
- **AND** top-1 accuracy MUST 不计为命中

#### Scenario: metric 口径记录
- **WHEN** 系统输出 BeamBench baseline metrics
- **THEN** 报告 MUST 声明 metric 是否使用官方非环形 DBA、64-beam circular DBA、beam shift 或其它配置
- **AND** 不同口径的数值 MUST 使用不同字段名或明确注释，避免混用

### Requirement: 复现报告与文档交付
系统 MUST 生成或维护 `README_REPRODUCE.md`、`ENVIRONMENT.md`、`DATASET_STRUCTURE.md`、`BASELINE_REPORT.md`、`PATCH_NOTES.md`、`TODO_FOR_ATTENTION_MODULE.md` 和 `results/reproduce_baseline.md`，用于记录复现步骤、环境、数据结构、baseline 结果、patch 和后续扩展位置。

#### Scenario: results 记录每次运行
- **WHEN** baseline 训练、评估或 mock smoke 完成
- **THEN** `results/reproduce_baseline.md` MUST 记录 command、当前仓库 git commit hash、官方 BeamBench commit hash、environment、dataset split、modalities、checkpoint path、metrics、日志路径和是否为 mock data

#### Scenario: README_REPRODUCE 给出命令
- **WHEN** 用户阅读 `README_REPRODUCE.md`
- **THEN** 文档 MUST 给出从环境检查、数据检查、mock smoke、真实数据评估到报告生成的逐步命令
- **AND** 所有项目相关 Python 命令 MUST 使用 `conda run -n kd_mm_beam`

#### Scenario: BASELINE_REPORT 不虚构结果
- **WHEN** 真实数据、官方权重、CUDA 或依赖不可用
- **THEN** `BASELINE_REPORT.md` MUST 明确记录阻塞点和下一步
- **AND** 文档 MUST NOT 填写伪造的论文结果、leaderboard 结果或真实数据指标

### Requirement: patch 与产物边界
系统 MUST 保持官方代码和本仓库代码改动可回滚、可解释。所有真实数据、训练输出、日志、缓存、checkpoint 和临时验证产物 MUST 遵守本仓库产物边界，默认不得纳入源码变更。

#### Scenario: 修改官方代码最小化
- **WHEN** 实现需要修改官方 BeamBench 代码或 vendored 代码
- **THEN** patch MUST 只覆盖运行所必需的最小范围
- **AND** `PATCH_NOTES.md` MUST 说明修改了什么、为什么修改、是否影响官方结果可比性和如何回滚

#### Scenario: 本地产物不提交
- **WHEN** baseline 运行生成 outputs、logs、cache、checkpoint、mock runtime files 或预测 CSV
- **THEN** 这些产物 MUST 位于 ignored 路径或被明确标记为本地产物
- **AND** 源码变更 MUST NOT 要求提交真实数据、训练输出或新生成 checkpoint

### Requirement: 后续 attention 模块插入点说明
系统 MUST 在 `TODO_FOR_ATTENTION_MODULE.md` 中明确说明后续 image/LiDAR 关键区域注意力、beam-guided attention 和 cross-attention fusion 最适合插入的代码位置。说明 MUST 包含文件、类、`forward` 函数或 batch 字段位置。

#### Scenario: image encoder 插入点
- **WHEN** 文档描述 image 关键区域注意力
- **THEN** 文档 MUST 指向 image encoder 输出 feature 或 token 的位置
- **AND** 文档 MUST 说明该位置如何连接到 BeamBench Camera AE + GPS 或本仓库 image fusion baseline

#### Scenario: LiDAR encoder 插入点
- **WHEN** 文档描述 LiDAR 关键区域注意力或 beam-guided attention
- **THEN** 文档 MUST 指向 LiDAR encoder 输出 BEV feature、global feature 或 BGAM mask/gate 前后的候选插入位置
- **AND** 文档 MUST 说明该位置是否保留二维空间维度供 attention 使用

#### Scenario: GPS embedding 与 fusion 插入点
- **WHEN** 文档描述 beam-guided attention 或 cross-attention fusion
- **THEN** 文档 MUST 指向 GPS embedding、late fusion concat、classifier head 和 `CLSTokenTransformerFusionNet.forward` 中模态 token 生成或 transformer fusion 的位置
- **AND** 文档 MUST 说明 dataloader batch 中是否能拿到历史 beam、GPS、scene id、timestamp 和 target label
