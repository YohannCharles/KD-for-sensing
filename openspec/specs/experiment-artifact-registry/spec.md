# experiment-artifact-registry Specification

## Purpose
定义训练/评估 artifact、checkpoint registry、sidecar metadata 和复现所需记录，确保模型权重、归一化工件、场景隔离 registry 与评估入口之间存在可审计、可复用且不依赖 legacy fallback 的解析契约。
## Requirements
### Requirement: 最佳 checkpoint 归档
训练流程 MUST 提供统一的最佳 checkpoint 归档能力。默认归档目录 MUST 为 `outputs/best_checkpoints/`，并且可通过配置覆盖。每次训练结束时，归档目录 MUST 至少保存当前配置在验证 Top-1 accuracy 上最高的 checkpoint；当训练过程中出现新的最高验证 Top-1 accuracy 时，系统 MAY 立即更新归档。

#### Scenario: strong 训练归档最高精度 checkpoint
- **WHEN** 用户运行 `configs/<modality>/strong.yaml` 或等价 strong 配置完成训练
- **THEN** 系统 MUST 将该配置验证 Top-1 accuracy 最高的 checkpoint 复制到归档目录
- **AND** 归档文件名 MUST 包含配置 slug、`strong` 和 `acc_<val_top1>`，例如 `<slug>_strong_acc_<val_top1>.pth`
- **AND** 原运行目录下的 checkpoint MUST 保留，不得被移动或删除

#### Scenario: 同一 slug 刷新最高精度
- **WHEN** 同一配置 slug 产生新的更高验证 Top-1 accuracy checkpoint
- **THEN** 归档目录 MUST 指向或保留该最高精度 checkpoint 作为默认候选
- **AND** 系统 MUST 避免默认解析到同一 slug 的旧低精度 checkpoint

### Requirement: checkpoint 解析优先级
评估权重解析 MUST 支持从最佳 checkpoint 归档目录加载匹配 checkpoint。显式传入的绝对路径或评估入口 `--weights` MUST 保持最高优先级；未显式指定时，系统 MAY 查找归档目录中的匹配 checkpoint。训练流程 MUST 不再为了 KD teacher 加载 checkpoint，且 MUST 不读取 `distillation.teacher_model_name`。

#### Scenario: 显式权重路径覆盖 registry
- **WHEN** 用户通过评估入口 `--weights` 或配置中的绝对路径显式指定 checkpoint
- **THEN** 系统 MUST 加载该显式路径
- **AND** 系统 MUST 不用归档目录中的候选替换该显式路径

#### Scenario: 训练不解析 KD teacher checkpoint
- **WHEN** 用户启动任一受支持训练配置
- **THEN** 训练流程 MUST 不调用 KD teacher checkpoint 解析
- **AND** 配置中若出现 `distillation.teacher_model_name` MUST 在配置解析阶段失败

### Requirement: 归档 metadata 与归一化工件关联
归档 checkpoint MUST 具备可机器读取的 metadata，用于记录源运行目录、配置 slug、模态、训练模式、epoch、验证 Top-1 accuracy、源 checkpoint 路径、split 信息和训练归一化工件路径。启用 GPS、LiDAR 或 mmWave 归一化时，metadata MUST 能让评估入口复用训练时的 scaler 或 normalizer/stats。

#### Scenario: 写入归档 sidecar
- **WHEN** 系统将 checkpoint 复制到归档目录
- **THEN** 系统 MUST 写入同名或可关联的 JSON sidecar metadata
- **AND** metadata MUST 记录验证 Top-1 accuracy、源 `run_dir`、源 checkpoint、配置 slug、训练模式和启用模态
- **AND** metadata MUST 不记录 KD 模式

### Requirement: 场景隔离的最佳 checkpoint registry
最佳 checkpoint registry MUST 按 DeepSense6G scene 或 scenegroup 隔离。默认 registry 目录 MUST 位于当前输出 scope 下，例如 `outputs/scene9/best_checkpoints/`、`outputs/scene31/best_checkpoints/`、`outputs/scene32/best_checkpoints/`、`outputs/scenegroup_s32_s34/best_checkpoints/` 和 `outputs/scenegroup_s31_s34/best_checkpoints/`。根级 `outputs/best_checkpoints/` MUST 只作为 legacy registry 输入由整理 manifest 审计，不得作为当前默认写入目标。

#### Scenario: Scenario 9 registry 写入 scene9
- **WHEN** 用户运行 Scenario 9 strong 训练并产生新的最高验证 Top-1 checkpoint
- **THEN** 系统 MUST 将归档 checkpoint 写入 `outputs/scene9/best_checkpoints/`
- **AND** metadata sidecar MUST 记录 `scene_id: 9` 和 `scene_slug: scene9`

#### Scenario: 默认 Scenario 31 registry 不复用其它场景
- **WHEN** 用户运行默认 Scenario 31 评估配置且未显式指定绝对 checkpoint
- **THEN** 系统 MUST 优先查找 `outputs/scene31/best_checkpoints/`
- **AND** 系统不得默认加载 `outputs/scene9/best_checkpoints/`、`outputs/scene32/best_checkpoints/` 或任一 scenegroup registry 中同 slug 的 checkpoint

#### Scenario: Scenario 32 registry 不复用 scene31
- **WHEN** 用户运行显式 Scenario 32 评估配置且未显式指定绝对 checkpoint
- **THEN** 系统 MUST 优先查找 `outputs/scene32/best_checkpoints/`
- **AND** 系统不得默认加载 `outputs/scene31/best_checkpoints/` 或任一 scenegroup registry 中同 slug 的 checkpoint

#### Scenario: 多场景 registry 写入 scenegroup
- **WHEN** 用户运行覆盖 scenes 32、33、34 的多场景训练并产生新的最高验证 Top-1 checkpoint
- **THEN** 系统 MUST 将归档 checkpoint 写入 `outputs/scenegroup_s32_s34/best_checkpoints/`
- **AND** metadata sidecar MUST 记录 scene scope 和参与的 train/validation/test scenes

#### Scenario: 多场景评估不回退到单场景 registry
- **WHEN** 用户运行多场景评估配置且未显式指定绝对 checkpoint
- **THEN** 系统 MUST 优先查找匹配 scenegroup 的 registry
- **AND** 系统不得默认加载 `outputs/scene31/best_checkpoints/` 中同 slug 的 checkpoint 作为替代

#### Scenario: legacy 根级 registry 进入整理复核
- **WHEN** 整理 manifest 扫描到 `outputs/best_checkpoints/`
- **THEN** manifest MUST 将其标记为 legacy registry
- **AND** 只有当 sidecar metadata 能唯一确定目标 scene 或 scenegroup 且目标无冲突时，manifest MAY 建议迁移到 canonical registry
- **AND** 否则 manifest MUST 标记为人工复核或 archive

#### Scenario: 显式绝对 checkpoint 仍最高优先级
- **WHEN** 用户通过绝对路径显式指定 teacher checkpoint 或评估权重
- **THEN** 系统 MUST 使用该显式路径
- **AND** scene 或 scenegroup registry 不得替换该路径

### Requirement: Teacher reliability registry artifact
实验产物体系 MUST 支持 teacher reliability registry。该 registry MUST 按场景隔离，引用 teacher checkpoint 和指标来源，并能被 Stage 2/3 配置稳定解析。

#### Scenario: 默认 Scene31 teacher registry 写入 scene31 输出组
- **WHEN** 用户为默认 Scenario 31 构建 teacher reliability registry
- **THEN** 默认输出路径 MUST 位于 `outputs/scene31/`
- **AND** registry MUST 记录 `scene_id: 31` 或等价 scene metadata

#### Scenario: Scene32 teacher registry 写入 scene32 输出组
- **WHEN** 用户为显式 Scenario 32 构建 teacher reliability registry
- **THEN** 默认输出路径 MUST 位于 `outputs/scene32/`
- **AND** registry MUST 记录 `scene_id: 32` 或等价 scene metadata

#### Scenario: registry 引用 checkpoint metadata
- **WHEN** teacher checkpoint 有 checkpoint registry sidecar metadata
- **THEN** teacher reliability registry MUST 记录 checkpoint 路径
- **AND** registry MUST 保留可追溯到源 run_dir、epoch 和验证 Top-1 的 metadata 或引用

#### Scenario: Stage 2 解析 registry 路径
- **WHEN** Stage 2 配置提供相对 teacher registry 路径
- **THEN** 系统 MUST 按项目根目录解析该路径
- **AND** 如果文件不存在，错误信息 MUST 包含解析后的绝对路径

### Requirement: Teacher metrics export
单模态 teacher 训练产物 MUST 提供 teacher registry 可读取的指标文件或等价 metadata。指标 MUST 至少包含模态、selected/best epoch、验证 Top-1、验证 Top-3、验证 Top-5、验证 ADBA、训练 Top-1、checkpoint 路径、checkpoint 来源、selection metric 和 selection mode。`best_epoch` MUST 指向 registry 默认应使用的 selected checkpoint epoch；如果系统同时保存最高 Top-1 checkpoint，指标 MUST 另行记录 Top-1 checkpoint 或 Top-1 epoch，不得用 Top-1 epoch 覆盖 objective-selected epoch。

#### Scenario: teacher 训练完成写出 metrics
- **WHEN** 单模态 teacher 训练完成至少一个 epoch
- **THEN** 输出目录 MUST 包含可供 registry 构建脚本读取的指标数据
- **AND** 指标数据 MUST 包含 `modality`、`best_epoch`、`val_acc_top1`、`val_acc_top3`、`val_acc_top5`、`val_adba` 和 `train_acc_top1`
- **AND** 指标数据 MUST 包含 `selection_metric`、`selection_mode`、`checkpoint` 或等价 checkpoint 路径字段
- **AND** `best_epoch` MUST 与 `checkpoint` 对应的 epoch 一致

#### Scenario: early stopping objective 作为默认 teacher 指标选择
- **WHEN** 训练流程保存了 early stopping objective 对应的 `best.pth` 且用户未显式要求 Top-1 teacher
- **THEN** teacher metrics MUST 使用 early stopping objective 对应的 epoch 作为 `best_epoch`
- **AND** teacher metrics MUST 将 checkpoint 路径记录为 `best.pth` 或其归档副本
- **AND** teacher metrics MUST 仍可记录最高 Top-1 epoch 作为附加诊断字段

#### Scenario: 显式 Top-1 teacher 指标选择
- **WHEN** 用户或配置显式要求按验证 Top-1 选择 teacher checkpoint
- **THEN** teacher metrics MAY 使用最高验证 Top-1 epoch 作为 `best_epoch`
- **AND** teacher metrics MUST 将 `selection_metric` 记录为 Top-1 accuracy
- **AND** teacher metrics MUST 将 checkpoint 路径记录为 `best_top1.pth` 或其归档副本

#### Scenario: metrics 与 checkpoint 模态不一致
- **WHEN** teacher metrics 中的 `modality` 与 registry 当前模态不一致
- **THEN** registry 构建流程 MUST 拒绝该输入
- **AND** 错误信息 MUST 包含期望模态和实际模态

### Requirement: Teacher metrics checkpoint objective selection
Teacher metrics helper MUST 按 metrics 或 checkpoint metadata 中声明的 selection objective 选择 checkpoint。未显式要求 Top-1 teacher 时，helper MUST 优先使用 objective-selected checkpoint；对 LiDAR teacher，helper MUST NOT 在存在 `best.pth` 或 objective checkpoint metadata 时默认选择 `best_top1.pth`。

#### Scenario: helper 使用 metrics 中声明的 checkpoint
- **WHEN** teacher metrics 或 checkpoint sidecar metadata 提供可访问的 checkpoint 路径和 selection metadata
- **THEN** teacher metrics helper MUST 使用该 checkpoint 路径
- **AND** helper MUST 记录 `selection_metric`、`selection_mode`、`selected_epoch` 和 checkpoint 来源

#### Scenario: LiDAR teacher 默认使用 objective checkpoint
- **WHEN** LiDAR teacher run 同时包含 `checkpoints/best.pth` 和 `checkpoints/best_top1.pth`，且用户未显式要求 Top-1 teacher
- **THEN** teacher metrics helper MUST 选择 `checkpoints/best.pth`
- **AND** helper MUST NOT 因为 `best_top1.pth` 存在而覆盖 `best.pth`

#### Scenario: 显式 Top-1 teacher 使用 best_top1 checkpoint
- **WHEN** 用户显式指定 checkpoint 路径为 `best_top1.pth` 或 registry selection metric 为验证 Top-1
- **THEN** teacher reliability registry MAY 选择 `best_top1.pth`
- **AND** registry MUST 将 checkpoint 来源标记为 explicit 或 top1-selection

#### Scenario: objective checkpoint 缺失时报错
- **WHEN** LiDAR teacher run 缺少 metrics checkpoint 路径和 `checkpoints/best.pth`，且用户未显式要求 Top-1 teacher
- **THEN** registry 构建流程 MUST 抛出清晰错误
- **AND** 错误信息 MUST 提示提供 objective checkpoint、重建 teacher metrics，或显式选择 Top-1 checkpoint

#### Scenario: registry 保留 Top-1 诊断指标
- **WHEN** registry 使用 objective checkpoint 而 run 同时存在 Top-1 指标
- **THEN** teacher reliability registry MUST 保留 objective checkpoint 的指标
- **AND** registry MAY 记录最高 Top-1 epoch 和 Top-1 value 作为诊断字段
- **AND** Stage 2/3 默认 teacher 加载 MUST 使用 objective checkpoint 路径

### Requirement: checkpoint 保留策略
训练和清理工作流 MUST 区分复现必需 checkpoint、选择指标 checkpoint、恢复训练 checkpoint 和临时 checkpoint。默认清理策略 MUST 保护 `best.pth`、`best_top1.pth`、checkpoint sidecar metadata、归一化 artifacts、metrics、最终配置和 startup summary；`last.pth` 或重复 probe checkpoint MAY 进入候选，但 MUST 记录风险等级和保留理由。

#### Scenario: 默认保护最佳 checkpoint
- **WHEN** 清理 manifest 扫描包含 `checkpoints/best.pth` 或 `checkpoints/best_top1.pth` 的 run
- **THEN** manifest MUST 默认将这些 checkpoint 标记为 protected
- **AND** manifest MUST 保留对应 sidecar metadata 的保护关系

#### Scenario: last checkpoint 可作为候选
- **WHEN** run 已完成且存在 `checkpoints/last.pth`，同时存在受保护的最佳 checkpoint 和 metrics
- **THEN** manifest MAY 将 `last.pth` 列为可删除候选
- **AND** manifest MUST 记录该候选不是默认复现 checkpoint

### Requirement: checkpoint retention metadata
运行产物摘要 MUST 能表达 checkpoint retention 决策所需 metadata。系统 MUST 记录 checkpoint 来源、选择指标、selected epoch、run 状态、是否有 sidecar、是否有归一化 artifact 引用和是否属于 registry 默认候选。

#### Scenario: retention 摘要包含选择信息
- **WHEN** run index 或清理 manifest 汇总 checkpoint
- **THEN** summary MUST 包含 checkpoint 来源和 selection metadata（如果可用）
- **AND** 缺失 metadata 时 MUST 记录缺失状态而不是推断为可删除

### Requirement: 退役研究线 checkpoint 可进入清理候选
checkpoint 保留策略 MUST 区分当前主线复现必需 artifact 和已退役研究线产物。退役 Hist/P3/V8/V9 run 中的 checkpoint MAY 进入清理候选，但 manifest MUST 记录 checkpoint 类型、sidecar 状态、run 状态和是否有保留理由。

#### Scenario: 退役 Hist checkpoint 候选记录完整
- **WHEN** 清理 manifest 扫描到退役 Hist run 中的 checkpoint
- **THEN** manifest MUST 记录 checkpoint 文件名、大小、是否为 `best.pth` 或 `last.pth`、是否有 sidecar metadata 和源 run 目录
- **AND** manifest MUST 不因 checkpoint 位于退役 run 中就绕过保护状态检查

#### Scenario: 当前主线 best checkpoint 默认保护
- **WHEN** 清理 manifest 扫描到当前主线 scene-level `best_checkpoints` 或当前主线 run 的 `best.pth`、`best_top1.pth`
- **THEN** manifest MUST 默认将其标记为 protected
- **AND** 删除阶段 MUST 跳过这些 protected checkpoint

### Requirement: 稳定实验工件输出记录
训练和评估流程 MUST 在最终配置、训练日志或测试报告中记录 checkpoint 解析与归档信息。记录内容 MUST 包含实际加载 checkpoint 路径、加载来源、registry 目录、归档 checkpoint 路径、验证 Top-1 accuracy、归一化工件路径和实际 split 样本数。

#### Scenario: 训练日志记录归档结果
- **WHEN** 一次训练完成并启用最佳 checkpoint 归档
- **THEN** `train_log.json` 或等价训练日志 MUST 记录 registry 目录和归档 checkpoint 路径
- **AND** 日志 MUST 记录用于归档命名的验证 Top-1 accuracy
- **AND** 日志 MUST 继续记录 train/test CSV 路径和样本数

#### Scenario: 评估报告记录权重来源
- **WHEN** 一次评估加载 checkpoint
- **THEN** `test_report.json` MUST 记录最终 checkpoint 路径
- **AND** 报告 MUST 记录 checkpoint 来源是显式路径、registry 还是旧路径回退

### Requirement: 默认实验 checkpoint 可被时间戳输出目录解耦
默认评估工作流 MUST 不依赖固定 `outputs/<run_name>/checkpoints/best.pth` 作为唯一权重来源。当固定 `run_name` 已存在导致新训练输出目录追加时间戳时，后续评估 MAY 通过 registry 找到对应配置的最高验证 Top-1 checkpoint。

#### Scenario: 时间戳输出被评估复用
- **WHEN** 训练因为目标运行目录已存在而写入带时间戳后缀的新运行目录
- **THEN** 训练完成后 registry MAY 保存该运行的最高验证 Top-1 checkpoint
- **AND** 后续评估 MUST 能显式指定或从 registry 解析该 checkpoint

#### Scenario: 旧 KD 权重路径不再作为训练 fallback
- **WHEN** 用户已有旧式 `paths.weights_dir / teacher_model_name` checkpoint 且 registry 没有匹配候选
- **THEN** 当前训练流程 MUST 不使用该路径加载 teacher
- **AND** 评估入口仍可通过 `--weights` 显式指定待评估 checkpoint

### Requirement: 表面积收敛保持实验 artifact 兼容
删除冗余配置、入口或文档后，当前保留的训练和评估 workflow MUST 继续保存完整运行 artifact。使用保留的 virtual/overlay 配置时，运行目录 MUST 记录足够信息用于复现，不得要求用户恢复已删除的实体 YAML。已退役的 CRAF、MARF、G2D 和 Multimodal-NF 配置不得由 virtual alias 接管。

#### Scenario: virtual 配置训练 artifact 完整
- **WHEN** 用户使用当前保留的 virtual/overlay 配置启动训练并完成 artifact 写出
- **THEN** 运行目录 MUST 包含完整 `final_config.yaml`、`resolved_config.yaml`、`train_log.json`、checkpoint metadata 和 split/runtime metadata
- **AND** 这些 artifact MUST 足以说明实际模型、数据、loss、训练参数和 checkpoint 来源

#### Scenario: 删除 fallback 入口不影响 console script
- **WHEN** 重复脚本 wrapper 被删除
- **THEN** 对应 console script 或 `python -m kd_sensing.cli.*` 入口 MUST 继续通过 `--help` 检查
- **AND** README 推荐命令 MUST 使用仍存在的入口

#### Scenario: 研究脚本不进入核心 workflow 兼容承诺
- **WHEN** 保留的研究脚本未声明为包内 CLI
- **THEN** 核心训练、评估、预处理和 manifest 导出 workflow MUST 不依赖该脚本
- **AND** 该脚本的输出产物 MUST 继续位于 `.gitignore` 覆盖路径或显式本地输出目录

#### Scenario: 退役配置不被兼容接管
- **WHEN** 用户引用已删除的 CRAF、MARF、G2D 或 Multimodal-NF 配置路径
- **THEN** 配置加载器 MUST 给出清晰缺失或退役错误
- **AND** 系统 MUST 不生成同名 virtual 配置

### Requirement: Artifact schema 拆分兼容
训练和评估相关模块拆分后，用户可见 artifact schema MUST 保持兼容。`final_config.yaml`、`resolved_config.yaml`、`train_log.json`、`training_outputs.npz`、`metrics.json`、checkpoint sidecar、teacher metrics 和 TensorBoard tag 的关键字段、路径和含义 MUST 不因内部模块移动而改变。

#### Scenario: 训练 artifact 字段保持
- **WHEN** 训练流程内部 writer、objective metadata 或 runtime metadata helper 被拆分
- **THEN** `final_config.yaml`、`train_log.json` 和 `metrics.json` 中既有公开字段 MUST 保持可用
- **AND** focused tests MUST 覆盖关键字段 presence

#### Scenario: objective metadata 拆分后兼容
- **WHEN** objective metadata 表、alias、history fields 或 TensorBoard schema 被迁移到窄模块
- **THEN** 训练、验证和评估 MUST 继续解析同一组 objective、metric alias、metric mode 和 history fields
- **AND** 现有 objective tests MUST 保持通过

### Requirement: 运行状态产物
训练和评估入口 MUST 尽量写出机器可读运行状态产物，使 run index 能判断启动、正常完成和 Python 异常失败。状态产物 MUST 保持轻量，并且 MUST 不改变现有 `final_config.yaml`、`resolved_config.yaml`、`metrics.json`、`train_log.json`、checkpoint 或 TensorBoard 语义。

#### Scenario: 训练启动写出状态
- **WHEN** 训练入口创建 run_dir 并完成初始配置解析
- **THEN** 系统 MUST 写出 `run_status.json` 或等价 runtime status 字段
- **AND** 状态 MUST 至少包含 `state: running`、run_dir、config path、start time、pid、experiment name、task、objective 和 enabled modalities

#### Scenario: 训练正常完成更新状态
- **WHEN** 训练完成并写出最终 metrics、train log 和 checkpoint metadata
- **THEN** 系统 MUST 将运行状态更新为 `complete`
- **AND** 状态 MUST 记录 end time、duration、primary metric、best checkpoint 和 metrics path

#### Scenario: Python 异常失败更新状态
- **WHEN** 训练或评估入口捕获到未处理 Python exception 并准备退出
- **THEN** 系统 SHOULD 将运行状态更新为 `failed`
- **AND** 状态 SHOULD 记录异常类型、异常消息和可查看的日志路径

#### Scenario: SIGKILL 无法捕获
- **WHEN** 训练进程被系统或用户以不可捕获方式终止
- **THEN** 系统 MAY 无法更新运行状态产物
- **AND** run index MUST 仍能通过日志和 partial artifacts 推断 killed、stale 或 partial 状态

### Requirement: Debug metrics logging
The training workflow MUST persist debug diagnostics in machine-readable run logs when debug mode is enabled. The diagnostics MUST be scoped so normal runs are unaffected when debug mode is disabled.

#### Scenario: 持久化首 batch 诊断
- **WHEN** CSI first-batch debug diagnostics are produced
- **THEN** the workflow MUST write them to the run log, metadata artifact or TensorBoard text/scalar stream
- **AND** the stored record MUST distinguish train and validation batch sources

#### Scenario: 持久化 epoch 训练健康指标
- **WHEN** epoch-level grad norm and param delta diagnostics are produced
- **THEN** the workflow MUST append them to the epoch metrics log
- **AND** normal training metrics arrays MUST remain backward compatible for existing analysis scripts

### Requirement: 入口输出边界显式
每个长期保留 CLI 或脚本入口 SHALL 有明确输出边界。入口 MUST 将训练、诊断、cache、checkpoint 和报告输出限定在 ignored 本地产物目录或显式用户指定目录，不得写入源码目录。

#### Scenario: 新诊断入口声明输出边界
- **WHEN** 新增 research diagnostic 或 benchmark CLI
- **THEN** maintainer context index 或 inventory MUST 记录默认输出目录和是否只读
- **AND** 输出 MUST 位于 `outputs/`、`logs/`、dataset preparation target 或显式本地路径边界内

### Requirement: Ledger provenance links
实验 artifact registry MUST 为 experiment ledger 提供可追溯的 checkpoint/run provenance。Ledger 记录 MAY 引用 registry sidecar，但 MUST 不复制真实 checkpoint。

#### Scenario: ledger 引用 checkpoint sidecar
- **WHEN** run 目录包含 checkpoint sidecar metadata
- **THEN** ledger 或 harvester summary MUST 记录 sidecar path、checkpoint path、selection metric、selected epoch、run_dir 和 config digest
- **AND** 缺少 sidecar 时 MUST 标记 provenance incomplete

#### Scenario: registry 与 claim candidate 关联
- **WHEN** claim candidate 引用某个 best checkpoint
- **THEN** candidate MUST 记录 checkpoint 来源是 explicit path、scene/scenegroup registry、run-local checkpoint 还是 unavailable
- **AND** 不明确来源 MUST 阻止 candidate 自动升级为 reviewed claim

