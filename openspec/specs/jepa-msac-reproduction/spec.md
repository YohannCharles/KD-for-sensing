# jepa-msac-reproduction Specification

## Purpose
定义 JEPA-MSAC 论文复现 workflow 的两阶段训练、数据协议、模型组件、指标、报告、ablation、产物边界和文档账本要求，使该复现能力能够以 paper/workflow baseline 的形式进入当前项目规范，同时保持真实数据、cache、checkpoint 和训练输出在本地或 ignored output 边界内。

## Requirements

### Requirement: JEPA-MSAC workflow 边界
系统 MUST 提供 JEPA-MSAC 论文复现 workflow，并将其标记为 paper/workflow baseline。该 workflow MUST 支持 Stage 1 JEPA-MSAC 预训练、Stage 2 冻结 backbone task-head 训练、evaluation/report 汇总，并 MUST 不作为普通 `modular_sequence` baseline 或 legacy 入口实现。

#### Scenario: 构建 JEPA-MSAC workflow 配置
- **WHEN** 用户加载 JEPA-MSAC smoke 或 paper-aligned 配置
- **THEN** 配置 MUST 记录 workflow family 为 `jepa_msac`
- **AND** 配置 MUST 声明 stage、DeepSense6G scene、window protocol、enabled modalities、target schema 和 output boundary
- **AND** 配置 MUST 不包含 `distillation.*`、teacher/student KD、HiST/Hist、standalone Top8、GPS residual 或 camera residual 字段

#### Scenario: 包内 CLI 可发现
- **WHEN** 开发者执行 `conda run -n kd_mm_beam kd-sensing-run-jepa-msac --help`
- **THEN** 命令 MUST 正常退出
- **AND** 帮助信息 MUST 包含 `--config`、`--stage`、`--dry-run`、`--pretrained-checkpoint` 或等价参数
- **AND** 入口 MUST 委托包内 `kd_sensing` workflow 实现，不得依赖新增 root-level 旧式训练脚本

### Requirement: 论文对齐数据协议
系统 MUST 为 JEPA-MSAC 构建可审计的数据协议，覆盖 DeepSense6G Scenario 32、13 帧 sliding window、`T_hist=8`、`T_pred=5`、64-beam codebook、Image/Radar/LiDAR/GPS/RF 历史输入、未来 localization/beam/RSSI targets 和 70/30 随机 train/test split。真实数据、cache 和 split 产物 MUST 保持在本地数据或 ignored output 边界内。

#### Scenario: 生成 Scenario 32 window manifest
- **WHEN** 用户运行 JEPA-MSAC manifest 或 dry-run audit
- **THEN** 系统 MUST 记录 scene、sample count、window length、history length、prediction length、split seed、train/test ratio、CSV 或 manifest 来源和启用模态
- **AND** 系统 MUST 不读取未启用模态的大数组
- **AND** 系统 MUST 不把生成 manifest、cache 或 split CSV 纳入源码目录

#### Scenario: 缺少论文必需字段时 blocked
- **WHEN** 本地 Scenario 32 数据缺少任一必需输入或 target 字段
- **THEN** workflow MUST 失败或标记为 blocked
- **AND** 错误或 report MUST 指出缺失字段、对应论文语义和可执行修复提示
- **AND** 系统 MUST 不用伪标签或 mock target 冒充真实复现

### Requirement: RF 历史字段映射
JEPA-MSAC workflow MUST 将论文 RF beam-level RSRP 历史表达为 workflow 专用 beam-power/RF-history 字段，并在 metadata 中记录其与仓库现有 mmWave/beam-power schema 的映射。系统 MUST 不把 `rf` 注册为新的 canonical modality。

#### Scenario: RF 历史输入进入 tokenizer
- **WHEN** batch 包含历史 beam-power 或 RSRP vector
- **THEN** workflow MUST 构造形状兼容 `[B, T_hist, K]` 或 `[B, T, K]` 的 RF history tensor
- **AND** RF tokenizer MUST 将每帧 RF vector 投影为一个 token
- **AND** runtime metadata MUST 记录 `paper_modality: RF`、仓库字段名、beam count 和 target/source split

#### Scenario: 拒绝 canonical rf modality
- **WHEN** 用户在通用 `modalities` 配置中声明 `rf`
- **THEN** 系统 MUST 拒绝该配置
- **AND** 错误信息 MUST 指向现有 canonical modality 和 JEPA-MSAC workflow-local RF mapping

### Requirement: 多模态 tokenization 与位置编码
JEPA-MSAC 模型 MUST 将 Image、Radar、LiDAR、GPS 和 RF 输入映射到统一 D 维 token space，并添加 factorized temporal、modality 和 intra-frame positional embeddings。默认 paper-aligned token counts MUST 支持 Image=9、Radar=16、LiDAR=16、GPS=1、RF=1，且 token counts、latent dimension 和 backbone depth MUST 可配置。

#### Scenario: tokenizer 输出统一 token schema
- **WHEN** 模型收到 batch size 为 `B`、总帧数为 `T` 的 paper-aligned 多模态输入
- **THEN** Image tokenizer MUST 输出 `[B, T, 9, D]`
- **AND** Radar tokenizer MUST 输出 `[B, T, 16, D]`
- **AND** LiDAR tokenizer MUST 输出 `[B, T, 16, D]`
- **AND** GPS 和 RF tokenizer MUST 分别输出 `[B, T, 1, D]`
- **AND** concat 后 token 序列 MUST 保留可追踪的 modality、time 和 intra-frame token index

#### Scenario: 位置编码维度校验
- **WHEN** token 序列使用 factorized positional embedding
- **THEN** temporal embedding MUST 覆盖总帧数 `T`
- **AND** modality embedding MUST 覆盖五类 paper modalities
- **AND** intra-frame embedding MUST 覆盖最大 token count
- **AND** 超出配置上限时系统 MUST 抛出包含实际 shape 和配置上限的清晰错误

### Requirement: Temporal block-masked JEPA 预训练
Stage 1 MUST 使用 temporal block-masked JEPA 预训练：每个模态独立采样连续时间块作为 masked tokens，context encoder 只接收 visible tokens，EMA target encoder 接收完整 token sequence，predictor 使用 mask token 预测完整 latent sequence，loss 只在 masked tokens 上计算 SmoothL1 或配置声明的等价 latent loss。

#### Scenario: temporal block mask 采样
- **WHEN** mask ratio 为 `rho` 且总帧数为 `T`
- **THEN** 每个模态 MUST 采样长度为 `floor(rho * T)` 的连续时间块
- **AND** 同一模态被选中帧内的全部 tokens MUST 一起进入 masked set
- **AND** `I_keep` 与 `I_mask` MUST 不重叠且覆盖完整 token index set

#### Scenario: EMA target encoder 与 masked loss
- **WHEN** Stage 1 完成一次 optimizer step
- **THEN** target encoder 参数 MUST 不接收梯度
- **AND** target latent MUST 从 autograd graph detach
- **AND** target encoder MUST 在 optimizer step 后按 EMA momentum 更新
- **AND** JEPA loss MUST 只在 `I_mask` 对应 latent 上计算并记录 mask ratio、EMA momentum 和 latent norm diagnostics

### Requirement: Frozen backbone future latent inference
Stage 2 MUST 冻结 Stage 1 backbone，并将历史 observation tokens 作为 keep tokens、未来 target slots 作为 masked tokens，通过 frozen context encoder 和 predictor 生成 predictive latent state `S_pred`。`S_pred` MUST 按 future frame 聚合为 `[B, T_pred, D]`，供所有 task heads 复用。

#### Scenario: Stage 2 backbone 冻结
- **WHEN** Stage 2 从 JEPA-MSAC checkpoint 构建模型
- **THEN** context encoder、target encoder 和 predictor 的参数 MUST 默认 `requires_grad=false`
- **AND** optimizer MUST 只包含启用的 task heads 和配置显式允许训练的轻量 adapter 参数
- **AND** runtime metadata MUST 记录 freeze policy、checkpoint path 和 trainable parameter summary

#### Scenario: future latent shape
- **WHEN** frozen inference 使用 `T_hist=8` 和 `T_pred=5`
- **THEN** 模型 MUST 对 5 个未来 frame 生成 predictive latent state
- **AND** `S_pred` 形状 MUST 为 `[B, 5, D]`
- **AND** 输出 MUST 保留可选诊断字段以追踪 future mask slots、pooling strategy 和使用的 target modalities

### Requirement: JEPA-MSAC task heads
Stage 2 MUST 提供 localization、beam prediction 和 RSSI prediction heads。Localization head MUST 支持 constant-velocity coarse estimate 加 residual correction；beam head MUST 可拼接 predicted localization 作为 auxiliary geometry；RSSI head MUST 支持 beam-power residual/profile regression 并输出 scalar RSSI 或配置声明的等价 link-strength target。

#### Scenario: localization head 输出和 loss
- **WHEN** `S_pred` 和历史 GPS/location 可用
- **THEN** localization head MUST 输出 `[B, T_pred, 2]` 的未来轨迹
- **AND** coarse estimate MUST 使用历史位置 constant-velocity 或配置声明的 bootstrap MLP
- **AND** localization loss MUST 使用 L1 或配置声明的等价回归 loss

#### Scenario: beam 与 RSSI heads 使用 localization-guided cascading
- **WHEN** localization-guided cascading 启用
- **THEN** beam head 和 RSSI head MUST 能接收 `[S_pred; predicted_location]` 融合表示
- **AND** beam head MUST 输出 `[B, T_pred, K]` logits
- **AND** RSSI head MUST 输出 `[B, T_pred, K]` beam-power profile 和/或 `[B, T_pred]` scalar RSSI
- **AND** 关闭 cascading 时 workflow MUST 在 metadata 中记录 `localization_guidance=false`

### Requirement: 论文指标与报告
JEPA-MSAC evaluation/report MUST 计算论文对齐指标：representation quality 的 RRankMe 和 RLDA，localization 的 ADE/FDE，beam prediction 的 Top-1、Top-3 和 L1-RSRP diff，RSSI prediction 的 RMSE/MAE。报告 MUST 同时支持 horizon-wise 和 aggregate summary。

#### Scenario: task metric summary
- **WHEN** evaluation 收到 predictions、targets 和 beam-power reference
- **THEN** report MUST 输出 ADE、FDE、Top-1、Top-3、L1-RSRP diff、RSSI RMSE 和 RSSI MAE
- **AND** 每个 metric MUST 记录方向、单位、horizon aggregation 口径和可用样本数
- **AND** 缺少 beam-power reference 时 L1-RSRP diff MUST 标记为 unavailable，而不是填充零值

#### Scenario: representation quality summary
- **WHEN** frozen backbone latent matrix 和增强视图 latent 可用
- **THEN** report MUST 计算 RRankMe 和 RLDA
- **AND** report MUST 记录 latent dimension、sample count、augmentation count 和数值稳定处理方式
- **AND** 增强视图不可用时 RLDA MUST 标记为 unavailable 并保留 RRankMe 结果

### Requirement: Ablation 与 baseline report
Workflow MUST 支持论文关键 ablation/report 维度的 manifest 化记录，包括 latent dimension、mask ratio、mask pattern、modality ablation、untrained/E2E/frozen-head 对照、localization auxiliary 开关和 missing-history 设置。未实际运行的 ablation MUST 只记录计划或 blocked 状态。

#### Scenario: ablation manifest
- **WHEN** 用户运行 `--stage report` 或 ablation dry-run
- **THEN** 系统 MUST 写出 ablation manifest
- **AND** manifest MUST 记录每个 row 的配置路径、checkpoint provenance、run status、metrics path、claim status 和 caveat
- **AND** 未运行 row MUST 不出现在可引用结果表中

#### Scenario: baseline 对照不复制通用训练循环
- **WHEN** workflow 增加 AR、GRU、Transformer 或 reconstruction/supervised 对照
- **THEN** 对照 MUST 复用现有模块化组件、evaluation helper 或 workflow-local thin orchestration
- **AND** 系统 MUST 不新增长期维护的重复 trainer 或 dataset parser

### Requirement: 运行产物与文档账本
JEPA-MSAC workflow MUST 将所有运行产物写入 ignored 的 `outputs/`、`logs/`、`outputs/cache/` 或用户显式本地路径。源码变更 MUST 只包含实现、配置、测试、OpenSpec 和文档账本摘要，不得包含真实 checkpoint、cache、metrics CSV、figures 或 TensorBoard event。

#### Scenario: 训练产物边界
- **WHEN** Stage 1 或 Stage 2 运行完成
- **THEN** checkpoint、final config、train log、metrics、predictions、tables 和 figures MUST 写入 ignored runtime output 目录
- **AND** final config 或 metadata MUST 记录 dataset manifest、stage、objective、freeze policy、checkpoint provenance 和 command override
- **AND** `git status --short` MUST 不因运行产物出现可提交的 checkpoint/cache/log 文件

#### Scenario: 文档和 claim 状态同步
- **WHEN** JEPA-MSAC workflow 作为 current 或 local-ready 复现入口加入项目
- **THEN** `docs/mainline_model_catalog.md`、`docs/experiment_protocols.md`、`docs/result_claims_registry.md`、`docs/experiment_matrix.md`、README 和 `docs/project_surface_inventory.md` MUST 同步记录入口、协议、结果状态、输出边界和 caveat
- **AND** 没有完成 paper-aligned 长训练前，claim status MUST 为 `unverified`、`mock/smoke`、`blocked` 或 `local-ready`，不得标记为 official reproduction

### Requirement: 验证与测试覆盖
JEPA-MSAC implementation MUST 提供 focused tests，覆盖 config load、CLI help、manifest audit、tokenizer shape、positional embedding shape、temporal block mask、EMA update、masked loss、frozen inference、task head outputs、metrics/report schema、runtime metadata 和文档/allowlist guard。项目相关 Python 测试 MUST 使用 `conda run -n kd_mm_beam`。

#### Scenario: synthetic smoke 不依赖真实数据
- **WHEN** 开发者运行 JEPA-MSAC focused smoke tests
- **THEN** 测试 MUST 使用 synthetic tensors 或 fixture manifest
- **AND** 测试 MUST 不读取真实 `dataset/` 数据
- **AND** 测试 MUST 不写入真实 checkpoint、cache 或训练输出到源码目录

#### Scenario: OpenSpec 与架构检查
- **WHEN** implementation 完成
- **THEN** 开发者 MUST 运行 `openspec validate reproduce-jepa-msac --strict`
- **AND** 涉及 CLI、registry、文档 allowlist 或 model forward 时 MUST 运行对应 focused tests 和架构边界测试
