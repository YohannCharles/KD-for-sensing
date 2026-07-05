# canonical-config-resolution Specification

## Purpose
定义 canonical/virtual 配置解析规则，确保实体 YAML、overlay recipe 和命令行覆盖产生一致可复现的最终配置。
## Requirements
### Requirement: 虚拟 canonical fusion 配置解析
系统 MUST 支持从 canonical fusion 配置路径生成配置，即使该路径在磁盘上没有实体 YAML 文件。可生成路径 MUST 仅限当前 supervised/adaptation 入口，例如 `configs/fusion/<slug>_strong.yaml`、`configs/fusion/<slug>_lightweight.yaml`、snapshot 或 active overlay recipe。系统 MUST 不再生成 `logits_kd`、`rkd` 或包含 `distillation` block 的配置。

#### Scenario: 加载 supervised canonical fusion 配置路径
- **WHEN** 用户加载 `configs/fusion/gps_mmwave_lightweight.yaml` 且该文件不存在
- **THEN** 系统 MUST 解析该路径并生成可用于训练、评估和测试的最终配置
- **AND** 最终配置的 `experiment.name` 和 `output.run_name` MUST 为 `gps_mmwave_lightweight`
- **AND** 最终配置的 `experiment.task` MUST 为 `fusion`
- **AND** 最终配置 MUST 不包含 `distillation` 配置块

#### Scenario: KD virtual path 被拒绝
- **WHEN** 用户加载 `configs/fusion/gps_mmwave_logits_kd.yaml` 或 `configs/fusion/gps_mmwave_rkd.yaml`
- **THEN** 系统 MUST 抛出清晰错误
- **AND** 系统 MUST 不生成配置或回退为 lightweight 配置

#### Scenario: 非 canonical 缺失文件不自动生成
- **WHEN** 用户加载 `configs/custom/missing.yaml` 或 `configs/fusion/not_a_canonical_name.yaml` 且该文件不存在
- **THEN** 系统 MUST 抛出清晰错误
- **AND** 系统 MUST 不把任意缺失 YAML 当作可生成配置

#### Scenario: 实体配置文件优先
- **WHEN** 用户加载一个磁盘上存在的 YAML 配置文件
- **THEN** 系统 MUST 按实体 YAML 内容加载配置
- **AND** 系统 MUST 不用同名虚拟 canonical 规则覆盖该实体文件

### Requirement: canonical fusion slug 命名校验
系统 MUST 使用固定模态优先级 `image > radar > gps > lidar > mmwave` 解析 fusion slug。slug MUST 由两个到五个不同合法模态组成；乱序、重复、未知模态和单模态 fusion slug MUST 被拒绝。

#### Scenario: 按 canonical 顺序解析 slug
- **WHEN** 用户加载 `configs/fusion/image_radar_gps_lidar_mmwave_lightweight.yaml`
- **THEN** 系统 MUST 将 slug 解析为 `["image", "radar", "gps", "lidar", "mmwave"]`
- **AND** `model.primary.modalities` MUST 使用相同顺序

#### Scenario: 拒绝乱序 slug
- **WHEN** 用户加载 `configs/fusion/mmwave_gps_lightweight.yaml`
- **THEN** 系统 MUST 拒绝该路径
- **AND** 错误信息 MUST 提示 canonical slug 为 `gps_mmwave`

#### Scenario: 拒绝重复模态 slug
- **WHEN** 用户加载 `configs/fusion/image_image_lightweight.yaml`
- **THEN** 系统 MUST 拒绝该路径
- **AND** 错误信息 MUST 指出 fusion slug 不能包含重复模态

#### Scenario: 拒绝未知模态 slug
- **WHEN** 用户加载 `configs/fusion/image_wifi_lightweight.yaml`
- **THEN** 系统 MUST 拒绝该路径
- **AND** 错误信息 MUST 包含非法模态名称 `wifi`

#### Scenario: 拒绝单模态 virtual fusion slug
- **WHEN** 用户加载 `configs/fusion/mmwave_lightweight.yaml`
- **THEN** 系统 MUST 拒绝该路径
- **AND** 错误信息 MUST 引导用户使用 `configs/mmwave/lightweight.yaml`

### Requirement: 生成配置语义
虚拟 canonical fusion 配置 MUST 生成当前 supervised/adaptation 语义，包括任务类型、模态启用字段、primary 模型配置、loss、训练参数和输出 run name。生成配置 MUST 不包含 teacher checkpoint 来源、distillation type、temperature、alpha 或 RKD 参数。

#### Scenario: 生成 strong fusion 配置
- **WHEN** 用户加载 `configs/fusion/gps_lidar_mmwave_strong.yaml`
- **THEN** 最终配置 MUST 不包含 `distillation`
- **AND** `model.primary.type` MUST 为 strong fusion 模型或明确命名的 strong baseline
- **AND** `model.primary.modalities` MUST 为 `["gps", "lidar", "mmwave"]`
- **AND** 配置 MUST 启用 GPS、LiDAR 和 mmWave 对应的数据及模型输入字段

#### Scenario: 生成 lightweight fusion 配置
- **WHEN** 用户加载 `configs/fusion/gps_lidar_mmwave_lightweight.yaml`
- **THEN** 最终配置 MUST 不包含 `distillation`
- **AND** `model.primary.type` MUST 为 `cls_token_transformer_fusion` 或当前默认 lightweight fusion 模型
- **AND** `model.primary.modalities` MUST 为 `["gps", "lidar", "mmwave"]`

#### Scenario: 保持 image+radar 兼容参数
- **WHEN** 用户加载 `configs/fusion/image_radar_lightweight.yaml`
- **THEN** 最终配置 MUST 保持 image+radar upstream 兼容参数
- **AND** primary fusion 模型 GRU MUST 为当前 lightweight 默认
- **AND** 训练流程 MUST 不要求 teacher checkpoint 来源

#### Scenario: 命令行覆盖应用在生成配置之后
- **WHEN** 用户加载虚拟 canonical fusion 配置并传入覆盖项 `training.epochs=1`
- **THEN** 最终配置 MUST 使用 `training.epochs: 1`
- **AND** 其它由 canonical 规则生成的字段 MUST 保持有效

### Requirement: canonical overlay recipe 化
canonical fusion 配置和 advanced overlay 生成 MUST 由可审查的 recipe/table 驱动。objective、当前保留的 advanced overlay 和通用 fusion overlay MUST 按职责拆分定义，`build_virtual_config()` 入口 MUST 只负责路径识别、recipe 查找和应用。已退役的 G2D、CRAF 和 MARF overlay MUST 不再作为可生成入口存在。

#### Scenario: 既有 canonical 路径生成语义不变
- **WHEN** 用户加载既有 virtual canonical fusion 路径
- **THEN** 系统 MUST 通过 recipe 生成与变更前等价的关键配置语义
- **AND** experiment name、task、modalities、primary model、loss、training 和 output run name MUST 保持兼容

#### Scenario: objective overlay recipe
- **WHEN** 用户加载 `configs/fusion/gps_mmwave_occlusion_supervised.yaml`
- **THEN** 系统 MUST 通过 objective recipe 生成 `experiment.objective: occlusion`
- **AND** recipe MUST 启用 occlusion target、occlusion head、objective loss 和对应 early stopping 默认值

#### Scenario: advanced overlay recipe 错误可诊断
- **WHEN** 用户加载未知 advanced overlay 路径
- **THEN** 系统 MUST 抛出清晰错误
- **AND** 错误信息 MUST 列出可用 overlay recipe 名称

### Requirement: config/io 职责收敛
配置加载流程 MUST 将 config source 解析、命令行 overlay、normalization pipeline、migration guard、dataset-specific rules 和 schema validation 拆到明确 helper。`config/io.py` MAY 作为入口协调这些 helper，但 MUST 不手写 canonical overlay、dataset 专属规则、removed feature guard 或 objective 专属默认表的主要实现。

#### Scenario: 加载实体配置
- **WHEN** 用户加载磁盘上存在的 YAML 配置
- **THEN** `config/io.py` MUST 读取实体文件并应用覆盖、默认补全和校验
- **AND** MUST 不进入 canonical recipe 生成路径
- **AND** 后续 normalization 和 validation MUST 通过明确 helper 执行

#### Scenario: 加载 virtual 配置
- **WHEN** 用户加载缺失但合法的 canonical virtual 配置路径
- **THEN** `config/io.py` MUST 调用 canonical recipe 入口获得基础配置
- **AND** 后续覆盖、默认补全和校验流程 MUST 与实体配置一致

#### Scenario: dataset 专属规则位于 dataset rule helper
- **WHEN** 用户加载 Raymobtime s008、DeepSense6G snapshot 或其它 dataset 专属配置
- **THEN** dataset 专属配置约束 MUST 由 dataset rule helper 或等价 validation helper 执行
- **AND** `config/io.py` MUST 不直接维护该 dataset 的完整业务规则

#### Scenario: migration guard 独立于 io 入口
- **WHEN** 用户配置已删除的 image motion profile、image cache、legacy encoder 或其它迁移拒绝项
- **THEN** 系统 MUST 由 migration guard 或等价 helper 抛出清晰错误
- **AND** `config/io.py` MUST 不直接维护所有已删除选项的完整拒绝逻辑

#### Scenario: normalization 顺序可测试
- **WHEN** 开发者对实体配置、virtual canonical 配置、snapshot 配置或 Raymobtime 配置运行 config load 等价测试
- **THEN** 测试 MUST 能验证 source、overlay、normalization 和 validation 的执行顺序
- **AND** 命令行覆盖 MUST 继续在配置生成后生效，并在必要的 runtime requirement 校验前被考虑

### Requirement: 高级配置矩阵优先使用 recipe
高级 fusion、objective 和当前保留的 ablation 配置矩阵 MUST 优先由 canonical recipe 或 overlay recipe 生成。实体 YAML MUST 只保留无法由 recipe 无损表达、需要人工编辑作为 base/example、或仍处于明确迁移窗口的配置。已退役的 G2D、CRAF 和 MARF 配置路径 MUST 不由 recipe 或 virtual alias 接管。

#### Scenario: 缺失高级 overlay 配置可生成
- **WHEN** 用户加载已声明支持的高级 fusion overlay 配置路径且磁盘上不存在实体 YAML
- **THEN** 配置加载器 MUST 通过 recipe 生成完整配置
- **AND** 训练、评估和 dry-run 工作流 MUST 像实体 YAML 一样消费该配置

#### Scenario: 保留实体 YAML 优先
- **WHEN** 用户加载磁盘上仍存在的高级 fusion YAML
- **THEN** 配置加载器 MUST 使用实体 YAML 内容
- **AND** 同名 recipe MUST 不覆盖用户在实体 YAML 中显式维护的字段

#### Scenario: 删除实体 YAML 后 final config 完整
- **WHEN** 用户通过 virtual/overlay 配置完成训练或 dry-run artifact 写出
- **THEN** `final_config.yaml` 和 `resolved_config.yaml` MUST 保存完整解析配置
- **AND** 运行产物 MUST 不依赖原始 YAML 文件继续存在

### Requirement: 可生成配置删除前必须有等价检查
删除实体配置前，项目 MUST 有 focused test 或脚本验证替代 virtual/overlay 配置的关键语义。关键语义 MUST 至少覆盖 experiment name、task、dataset type、enabled modalities、model type、loss type、training schedule、output run name 和 checkpoint 来源。

#### Scenario: 关键字段等价
- **WHEN** 开发者准备删除一个可生成实体 YAML
- **THEN** 测试 MUST 比较删除前实体配置和 recipe 生成配置的关键字段
- **AND** 允许差异 MUST 在测试或设计文档中显式列出

#### Scenario: 非 canonical 缺失文件仍被拒绝
- **WHEN** 用户加载未声明 recipe 的缺失 YAML 路径
- **THEN** 系统 MUST 抛出清晰 `FileNotFoundError` 或迁移错误
- **AND** 系统 MUST 不把任意缺失 YAML 自动当作高级 overlay 配置

#### Scenario: 配置矩阵不重新实体化
- **WHEN** 开发者运行配置表面积回归检查
- **THEN** 检查 MUST 拒绝新增与已支持 recipe 等价的实体 YAML
- **AND** 如需新增实体 YAML，必须在 OpenSpec 中说明不能由 recipe 表达的字段

### Requirement: 高级实体 YAML 删除前必须通过等价检查
删除高级实体 YAML 前，项目 MUST 提供 focused test 或脚本比较实体配置和替代 virtual/overlay 配置的关键语义。关键语义 MUST 至少覆盖 experiment name、task、dataset type、enabled modalities、model type、loss type、training schedule、output run name 和 checkpoint 来源。

#### Scenario: 可生成高级配置关键字段等价
- **WHEN** 开发者删除一个由 recipe 覆盖的当前支持 token transformer、CSI/GPS/mmWave 组合或 ablation 实体 YAML
- **THEN** 等价检查 MUST 证明替代 virtual/overlay 配置的关键字段与原实体 YAML 一致
- **AND** 允许差异 MUST 在测试断言或设计文档中显式列出

#### Scenario: 退役配置不被 recipe 接管
- **WHEN** 被删除实体 YAML 属于 G2D、CRAF 或 MARF
- **THEN** 配置加载器 MUST 报告缺失或退役错误
- **AND** 系统 MUST 不生成同名 virtual 配置

#### Scenario: 删除后运行产物保存完整配置
- **WHEN** 用户使用已删除实体 YAML 对应的 virtual/overlay 路径启动 dry-run、训练或评估
- **THEN** 配置加载器 MUST 生成完整最终配置
- **AND** 运行目录中的 `final_config.yaml` 和 `resolved_config.yaml` MUST 不依赖原实体 YAML 继续存在

### Requirement: 高级 overlay recipe 必须按领域拆分
高级配置生成 MUST 将 objective、CSI hardening 和当前保留组合实验 overlay 的主要字段定义放入可审查的 recipe/table 或领域 helper 中。`build_virtual_config()` 和路径识别入口 MUST 只负责识别路径、查找 recipe 和应用 overlay。

#### Scenario: 新增当前支持 ablation 不扩写路径入口
- **WHEN** 开发者新增一个当前支持方法的 ablation 配置 overlay
- **THEN** 主要变更 MUST 位于对应 recipe/table 或领域 helper
- **AND** 不得在 `build_virtual_config()` 中新增大段方法专属字段表

#### Scenario: 新增 CSI 组合配置有明确 recipe 来源
- **WHEN** 开发者新增 CSI/GPS/mmWave 或 CSI hardening 组合配置的 virtual/overlay 支持
- **THEN** recipe MUST 明确声明模态集合、dataset 字段、模型类型、loss、training 和 output run name
- **AND** 非声明路径 MUST 继续抛出清晰缺失配置错误

### Requirement: 可生成配置不得重新实体化
当某类配置已经由 canonical recipe 或 advanced overlay 无损生成后，项目 MUST 防止等价实体 YAML 重新进入源码表面积。确需新增实体 YAML 时，OpenSpec change MUST 说明 recipe 无法表达的字段或人工样例用途。

#### Scenario: 表面积检查拒绝已支持 recipe 的实体 YAML
- **WHEN** 开发者新增与已支持 virtual/overlay recipe 等价的实体 YAML
- **THEN** 表面积回归检查 MUST 拒绝该文件
- **AND** 错误或测试说明 MUST 指向对应 recipe 路径或要求补充 OpenSpec 保留理由

#### Scenario: 人工样例配置保留原因可审计
- **WHEN** 项目保留一个不能删除的高级实体 YAML
- **THEN** inventory MUST 记录它作为 base、example、迁移窗口或不可 recipe 化实验的用途
- **AND** 后续删除或 recipe 化该文件 MUST 更新对应记录

### Requirement: 可生成配置必须有等价验证
实体 YAML 被 recipe、overlay 或 manifest generator 替代前，项目 MUST 用 focused test 或脚本验证关键解析语义。关键语义 MUST 至少覆盖 experiment name、task/objective、dataset type、enabled modalities、model type、loss type、training defaults、output run name 和 checkpoint/artifact policy。

#### Scenario: 删除可 recipe 化 YAML
- **WHEN** 某个实体 YAML 被分类为 recipe 可无损生成
- **THEN** config load focused test MUST 证明生成配置的关键字段与实体配置等价或列出允许差异
- **AND** 删除后 `final_config.yaml` 和 `resolved_config.yaml` MUST 仍能保存完整解析配置

#### Scenario: 非声明路径仍拒绝
- **WHEN** 用户加载未声明 recipe/overlay 的缺失 YAML
- **THEN** 配置加载 MUST 抛出清晰 `FileNotFoundError` 或 retired-route 错误
- **AND** 系统 MUST 不把任意缺失实验 YAML 自动接管为 virtual config

### Requirement: 重复小工具使用单一 config owner
配置 recipe、CLI overlay 和 model summary 需要 deep merge 时 MUST 使用单一 owner helper。项目 MUST 不保留多个行为近似但 copy 语义不同的 `deep_merge` 实现。

#### Scenario: 删除 recipe deep merge 副本
- **WHEN** canonical recipe 需要 deep merge
- **THEN** 代码 MUST 使用 `kd_sensing.config.io.deep_merge` 或迁移后的单一 owner
- **AND** `kd_sensing.config.canonical_recipes.common.deep_merge` MAY 被删除

### Requirement: YAML 解析使用项目依赖
配置加载 MUST 使用项目声明的 YAML 依赖解析 YAML。系统 MUST 不维护手写 YAML 子集 parser 或 optional-yaml fallback 作为当前配置语义的一部分。

#### Scenario: 加载 YAML 配置
- **WHEN** 用户加载实体 YAML、virtual config base 或 manifest YAML
- **THEN** 系统 MUST 使用 `pyyaml` 安全解析配置
- **AND** 解析结果 MUST 继续支持当前实体 YAML、`_base_`、命令行 override 和 config normalization 流程

#### Scenario: 删除手写 YAML fallback
- **WHEN** 项目环境缺失 `pyyaml`
- **THEN** 配置加载 MUST 失败并暴露依赖缺失
- **AND** 系统 MUST 不回退到手写 YAML 子集 parser

### Requirement: Canonical recipe 小层可合并
只包装少量常量表、没有独立 public API、没有多个真实实现且只被 `canonical.py` 消费的 recipe/dataclass 文件 MAY 合并到 `canonical.py` 或单一 owner。合并 MUST 保持 virtual config 关键语义、实体 YAML 优先和命令行覆盖顺序。

#### Scenario: 合并 fusion training recipe
- **WHEN** canonical fusion training defaults 从独立 dataclass 文件迁入 owner
- **THEN** `training_overrides()` 或等价逻辑 MUST 对 strong/lightweight、image-radar 和一般 fusion 生成与变更前一致的关键字段
- **AND** `tests/test_config_load_characterization.py` 或等价 focused test MUST 覆盖该行为

#### Scenario: 合并 objective overlay recipe
- **WHEN** objective overlay recipe 常量迁入 owner
- **THEN** occlusion、position 和 multitask virtual config MUST 保持 objective、dataset target、loss、training metric 和 output run name 语义
- **AND** 未知 overlay MUST 继续给出清晰错误

### Requirement: 配置瘦身不重新实体化
删除手写 parser 或合并 recipe 小层时，项目 MUST 不通过新增实体 YAML 或复制默认表来恢复同一复杂度。

#### Scenario: 不新增重复实体配置
- **WHEN** recipe 小层被合并
- **THEN** 实现 MUST 不新增与 recipe 等价的实体 YAML 来弥补删除
- **AND** final/resolved config MUST 继续保存完整解析结果

### Requirement: 虚拟 canonical 配置工作流
训练、评估和测试工作流 MUST 接受由配置加载器生成的虚拟 canonical fusion 配置。虚拟配置 MUST 在进入训练、评估、dry-run、override 合并、验证和 artifact 写出之前被解析为完整配置字典。

#### Scenario: 训练入口使用虚拟 canonical 配置
- **WHEN** 用户运行 `kd-sensing-train --config configs/fusion/gps_mmwave_lightweight.yaml`
- **THEN** 系统 MUST 解析该 canonical path 并启动 fusion lightweight 训练流程
- **AND** 训练流程 MUST 不要求 `configs/fusion/gps_mmwave_lightweight.yaml` 在磁盘上存在

#### Scenario: 评估入口使用虚拟 canonical 配置
- **WHEN** 用户运行 `kd-sensing-evaluate --config configs/fusion/gps_mmwave_lightweight.yaml --weights <path>`
- **THEN** 系统 MUST 解析该 canonical path 并构建对应 fusion primary 模型
- **AND** 评估流程 MUST 只准备该配置启用的模态输入

#### Scenario: dry-run 使用虚拟 canonical 配置
- **WHEN** 用户运行 `kd-sensing-train --config configs/fusion/gps_mmwave_lightweight.yaml --dry-run`
- **THEN** 系统 MUST 先生成 canonical 配置，再应用 dry-run 覆盖
- **AND** dry-run MUST 使用 synthetic dataset、单 epoch 和关闭 worker 的现有行为

#### Scenario: 保存完整 final config
- **WHEN** 使用虚拟 canonical 配置完成训练
- **THEN** 系统 MUST 在运行目录保存完整解析后的 `final_config.yaml`
- **AND** `final_config.yaml` MUST 包含训练复现所需的全部字段，而不是只保存虚拟路径或生成规则

#### Scenario: CLI override 覆盖虚拟配置
- **WHEN** 用户通过 `--override` 或点式未知参数覆盖虚拟 canonical 配置字段
- **THEN** 系统 MUST 在生成 canonical 配置之后应用这些覆盖
- **AND** 覆盖优先级 MUST 与实体 YAML 配置保持一致

### Requirement: 删除实体配置后 workflow 必须可复现
当当前保留的实体 YAML 被 recipe/overlay 替代后，训练和评估 workflow MUST 继续保存足够的 resolved/final 配置、运行元数据和 checkpoint 来源信息，保证不恢复被删除 YAML 也能理解实际运行参数。已退役的 CRAF、MARF、G2D 和 Multimodal-NF 实体 YAML 删除后 MUST 不提供同名 recipe/overlay 兼容。

#### Scenario: virtual 配置训练记录完整
- **WHEN** 用户使用当前保留的 virtual/overlay 配置完成训练或 dry-run artifact 写出
- **THEN** 运行目录 MUST 包含完整 `final_config.yaml`、`resolved_config.yaml`、训练元数据和 checkpoint 来源信息
- **AND** 这些 artifact MUST 能说明实际模型、数据、loss、训练参数和输出 run name

#### Scenario: 删除 YAML 不影响评估入口
- **WHEN** 某个当前保留的实体 YAML 被删除但对应 virtual/overlay 配置仍被声明支持
- **THEN** `kd-sensing-evaluate --config <deleted-yaml-path>` MUST 通过配置加载器解析等价最终配置
- **AND** 如果该路径未被声明支持，系统 MUST 抛出清晰缺失配置错误

#### Scenario: 退役 YAML 不支持 virtual fallback
- **WHEN** 被删除 YAML 属于 CRAF、MARF、G2D 或 Multimodal-NF
- **THEN** 系统 MUST 将其视为不支持路径
- **AND** 系统 MUST 不为其提供 virtual fallback

### Requirement: Baseline clone config diff artifact
The experiment workflow MUST support comparing a generated baseline clone against a reference baseline resolved config. The diff MUST separate allowed run identity differences from key behavior differences.

#### Scenario: 生成 A0 clone diff
- **WHEN** both `A0_original` and `A0_clone_generated` resolved configs are available
- **THEN** the workflow MUST produce a diff artifact comparing them
- **AND** the diff MUST ignore only allowlisted run identity fields such as run name, output directory, timestamp and seed when configured

#### Scenario: 关键字段差异失败
- **WHEN** the diff finds a difference in optimizer, scheduler, loss, dataset split, normalization, train RMS path, `seq_len`, `num_pred`, `num_classes`, model type, CSI encoder, representation core or beam head
- **THEN** the workflow MUST mark the parity check as failed
- **AND** the failure message MUST list the differing config paths

### Requirement: config/io 不承载业务规则实现
`kd_sensing.config.io` MUST 保持配置入口协调职责，负责加载实体 YAML 或 virtual config、应用命令行覆盖、调用 normalization pipeline 和调用 validation pipeline。objective 默认补全、模态推导、dataset-specific rules、迁移拒绝和 schema validation 的主要实现 MUST 位于独立 helper。

#### Scenario: Raymobtime 退役规则不写在 io 入口
- **WHEN** 开发者调整 Raymobtime s008 退役配置、旧 dataset 名称或旧 preprocessor 名称的拒绝规则
- **THEN** 主要实现 MUST 位于 migration guard、config validation helper 或 registry 拒绝 helper
- **AND** `config/io.py` MUST 只调用该 helper，不得恢复 Raymobtime dataset/preprocessor 运行路径

#### Scenario: removed image motion guard 不写在 io 入口
- **WHEN** 开发者调整已删除 image motion profile、cache 或 encoder 的拒绝逻辑
- **THEN** 主要实现 MUST 位于 migration guard 或 image profile validation helper
- **AND** `config/io.py` MUST 不直接维护该迁移规则的完整实现

#### Scenario: objective 默认值不写在 io 入口
- **WHEN** 开发者新增或调整 prediction objective 的默认 early stopping metric、loss weights 或 required target/head
- **THEN** 主要实现 MUST 位于 objective metadata、normalization helper 或 validation helper
- **AND** `config/io.py` MUST 不维护 objective 专属分支表

### Requirement: Fusion canonical 多模态配置矩阵
项目 MUST 为 `image`、`radar`、`gps`、`lidar`、`mmwave` 的所有必要多模态组合提供 canonical fusion supervised 配置矩阵。多模态组合 MUST 覆盖全部 10 个双模态组合、10 个三模态组合、5 个四模态组合和 1 个五模态组合。每个组合 MUST 提供可加载的 strong 和 lightweight canonical 配置路径；这些 canonical 配置 MAY 由 loader 生成，不要求每个路径都有实体 YAML 文件。

#### Scenario: 双模态 fusion 组合完整
- **WHEN** 开发者加载 `configs/fusion/<slug>_<mode>.yaml`
- **THEN** 系统 MUST 为所有合法双模态 slug 提供 `<slug>_strong.yaml` 和 `<slug>_lightweight.yaml`
- **AND** 系统 MUST 不要求提供 `<slug>_logits_kd.yaml` 或 `<slug>_rkd.yaml`

#### Scenario: 五模态 fusion 组合完整
- **WHEN** 开发者加载五模态 fusion canonical 配置
- **THEN** 系统 MUST 提供可加载的 `image_radar_gps_lidar_mmwave_strong.yaml` 和 `image_radar_gps_lidar_mmwave_lightweight.yaml`
- **AND** 系统 MUST 拒绝同 slug 的 KD 配置路径

### Requirement: Fusion canonical 配置语义
每个 canonical fusion 配置 MUST 使用固定模态顺序 `image`、`radar`、`gps`、`lidar`、`mmwave` 生成 slug。推荐/default fusion lightweight 路线 MUST 使用 `cls_token_transformer_fusion` 或当前 active fusion model。canonical 配置语义 MUST 不依赖实体 YAML 文件是否存在，且 MUST 不包含 distillation 或 frozen teacher runtime。

#### Scenario: fusion strong 配置
- **WHEN** 开发者加载 `configs/fusion/<slug>_strong.yaml`
- **THEN** 配置 MUST 设置 `experiment.task: fusion`
- **AND** 配置 MUST 将 `model.primary` 配置为 strong fusion baseline
- **AND** primary model modalities MUST 等于 slug 表示的模态集合

#### Scenario: fusion lightweight 默认配置
- **WHEN** 开发者加载 `configs/fusion/<slug>_lightweight.yaml`
- **THEN** 配置 MUST 设置 `experiment.task: fusion`
- **AND** 配置 MUST 将 `model.primary` 配置为 `cls_token_transformer_fusion` 或当前推荐 lightweight fusion 模型
- **AND** 配置 MUST 不构建 frozen teacher

### Requirement: 当前高级 fusion overlay 边界
当前保留的高级 fusion overlay MUST 只覆盖已批准的 objective-aware 或调试入口。已退役的 CRAF、MARF、G2D 和相关 ablation 配置 MUST 不再由 overlay recipe 或 virtual alias 生成。

#### Scenario: 保留实体 YAML 优先
- **WHEN** 用户加载一个仍存在的 `configs/fusion/*.yaml` 文件
- **THEN** 系统 MUST 使用该实体 YAML 的内容
- **AND** 不得用 virtual overlay 规则覆盖实体 YAML 中显式配置的字段

#### Scenario: 退役实体 YAML 不被 virtual 接管
- **WHEN** 被删除 YAML 属于 CRAF、MARF 或 G2D
- **THEN** 系统 MUST 将其视为不支持路径
- **AND** 系统 MUST 不为其提供 virtual fallback

#### Scenario: 配置矩阵测试覆盖 overlay
- **WHEN** 开发者运行 fusion 配置矩阵测试
- **THEN** 测试 MUST 覆盖当前保留 overlay 入口的可加载性和关键字段
- **AND** 测试 MUST 验证仍保留的实体 YAML 按兼容语义加载

### Requirement: 单模态 canonical 配置矩阵
项目 MUST 为每个受支持单模态 `image`、`radar`、`gps`、`lidar` 和 `mmwave` 提供统一命名的 canonical 配置矩阵。每个单模态目录 MUST 包含 `strong.yaml`、`lightweight.yaml` 和 `supervised.yaml`。canonical 配置 MUST 使用 `model.primary`、统一训练、验证、评估、loss、optimizer、scheduler、checkpoint 和输出目录语义。

#### Scenario: 单模态 strong 配置
- **WHEN** 开发者加载 `configs/<modality>/strong.yaml`
- **THEN** 配置 MUST 使用该模态对应的 `experiment.task`
- **AND** 配置 MUST 不包含 `distillation`
- **AND** 配置 MUST 将被训练主模型配置为对应 `<modality>_strong`
- **AND** 配置的 `experiment.name` 和 `output.run_name` MUST 使用 `<modality>_strong`

#### Scenario: 单模态 lightweight 配置
- **WHEN** 开发者加载 `configs/<modality>/lightweight.yaml`
- **THEN** 配置 MUST 使用该模态对应的 `experiment.task`
- **AND** 配置 MUST 不包含 `distillation`
- **AND** 配置 MUST 将被训练主模型配置为对应 `<modality>_lightweight`
- **AND** 配置的 `experiment.name` 和 `output.run_name` MUST 使用 `<modality>_lightweight`

#### Scenario: 单模态 supervised 配置
- **WHEN** 开发者加载 `configs/<modality>/supervised.yaml`
- **THEN** 配置 MUST 使用该模态对应的 `experiment.task`
- **AND** 配置 MUST 不包含 `distillation`
- **AND** 配置 MUST 将被训练主模型配置为明确的 supervised baseline

#### Scenario: 旧单模态 KD 配置被拒绝
- **WHEN** 开发者加载 `configs/<modality>/logits_kd.yaml` 或 `configs/<modality>/rkd.yaml`
- **THEN** 配置加载 MUST 失败
- **AND** 错误信息 MUST 指向 strong、lightweight 或 supervised 入口

### Requirement: canonical 配置命名与输出目录一致
canonical 配置 MUST 使用可预测的实验名和 run name。默认路径 MUST 便于用户按 strong/lightweight/supervised 或当前 workflow 顺序运行实验，并 MUST 支持命令行覆盖。

#### Scenario: canonical run name 与文件语义一致
- **WHEN** 开发者加载任意 canonical 配置
- **THEN** `experiment.name` MUST 与不含 `.yaml` 的文件 stem 一致
- **AND** `output.run_name` MUST 与 `experiment.name` 一致

#### Scenario: canonical 配置不解析 teacher checkpoint
- **WHEN** 用户加载当前 canonical 配置
- **THEN** 系统 MUST 不解析 teacher checkpoint
- **AND** 训练流程 MUST 只构建 primary model

#### Scenario: canonical KD checkpoint override 被拒绝
- **WHEN** 用户通过命令行覆盖 `distillation.teacher_model_name`
- **THEN** 配置加载 MUST 失败
- **AND** 错误信息 MUST 指向当前配置入口

### Requirement: Config surface distinguishes canonical, recipe, generated, and local/manual
配置生命周期 MUST 区分 canonical/current entity YAML、virtual config、recipe/generated config、experiment reproduction/local manual overlay、diagnostics manifest 和 retired config。可生成或本地队列型配置 SHOULD 通过 recipe/manifest/generator 表达；tracked entity YAML MUST 有 current 入口、复现实验、diagnostics manifest 或 local/manual 登记理由。

#### Scenario: 生成型配置不无限实体化
- **WHEN** 新增 Scene31 seed sweep、night-grid、next-round、ablation matrix 或其它规则化配置族
- **THEN** 项目 MUST 优先提供 recipe、manifest 或 generator sanity test
- **AND** 若提交实体 YAML，inventory 或 tasks MUST 说明为何不能只由 recipe 生成，以及该 YAML 的 lifecycle

#### Scenario: Canonical 配置保留实体入口
- **WHEN** 配置属于 README/current docs 推荐的 canonical single-modality、fusion、diagnostic 或 paper/workflow reproduction 入口
- **THEN** 实体 YAML MAY 保留
- **AND** virtual config 或 recipe MUST 不接管 retired KD、BGAM、viewer、Hist、Raymobtime、AMR mock 或 JEPA-MSAC 路径

### Requirement: Generated config recipes preserve resolved semantics
Recipe/generated config MUST 生成与等价实体 YAML 相同的 resolved config 语义，并在 sanity tests 中覆盖 run name、seed、epoch、sampler、loss weights、missing pattern、difficulty profile 和 output boundary 等关键字段。

#### Scenario: Recipe sanity validation
- **WHEN** generator 创建本地实验矩阵
- **THEN** focused tests MUST 校验 manifest 行、文件名/run name 和 resolved config 关键字段一致
- **AND** generator MUST 不写入 `outputs/`、`logs/`、checkpoint 或真实训练结果

### Requirement: Config cleanup keeps migration guards
配置表面瘦身 MAY 删除重复实体 YAML、旧 alias、未登记 local queue config 或退役路径，但 MUST 保留仍有当前迁移价值的 guard、错误信息或 retired summary。

#### Scenario: 删除重复配置
- **WHEN** 一个实体 YAML 可由 current recipe/virtual config 无损生成，且不属于 canonical/current 推荐入口
- **THEN** 本 change MAY 删除该 YAML 或迁到 local/manual generated surface
- **AND** 配置加载器 MUST 继续拒绝 retired config path，不能把旧路径静默映射到新 recipe

### Requirement: Generated experiment config families do not require entity YAML
规则化实验矩阵、seed sweep、night-grid、next-round queue 或本地 GPU 队列配置 MUST 优先由 base config、manifest 和 generator 表达。若实体 YAML 可由 generator 无损重建且不属于 canonical/current、paper/workflow reproduction 或 diagnostics manifest，它 MUST 从长期源码表面删除。

#### Scenario: Removable generated YAML
- **WHEN** generator 能重建实体 YAML 的 run name、seed、epoch、sampler、loss weights、missing pattern、dataset split、output boundary 和 critical overrides
- **THEN** 项目 MUST 保留 generator/manifest/base config
- **AND** 对应实体 YAML MAY 删除

#### Scenario: Retired path not regenerated
- **WHEN** generator 或 virtual config 解析规则处理实验矩阵
- **THEN** 它 MUST 不生成或接管 legacy KD、HiST/Hist、BGAM、viewer、Raymobtime、AMR-Net_gps_image 或 JEPA-MSAC retired path
- **AND** config migration guard MUST 继续 fail fast

### Requirement: Config generator has a small sanity check
保留的 config generator MUST 有 focused sanity check，覆盖 manifest 行、文件名/run name 和关键 resolved config 字段，且 MUST 不写入 `outputs/`、`logs/`、checkpoint 或真实训练结果。

#### Scenario: Generator sanity
- **WHEN** generator 更新 tracked config family 或 manifest
- **THEN** focused tests MUST 验证生成结果的核心语义
- **AND** 测试 MUST 使用临时目录或受控 config 输出路径

### Requirement: Canonical config 解析必须拆分 recipe 与 migration guard
Canonical config 重构 MUST 将 virtual recipe generation、overlay resolution、path alias handling 和 retired-route migration guards 保持为独立职责，并保持 load error 兼容。

#### Scenario: retired route 不被 virtual config 接管
- **WHEN** user loads a retired config path or retired KD alias
- **THEN** config loading MUST 按既有 migration guard 语义 fast fail
- **AND** virtual config generation MUST NOT create replacement configs for retired research lines

### Requirement: Config list 和 doctor
项目 MUST 提供 config list/doctor 能力，用于按 config family、lifecycle、formal/smoke/local/manual、是否需要真实数据、默认输出边界和 focused validation 对 tracked YAML 与 virtual config route 进行只读分类。Config doctor MUST 不将退役路线恢复为 virtual alias 或实体 YAML。

#### Scenario: 列出当前 config family
- **WHEN** 开发者运行 config list
- **THEN** 输出 MUST 按 canonical root、experiment family、diagnostics、preprocess、local/manual、baseline reproduction 和 retired guard 分类
- **AND** 每条输出 MUST 指向对应 README/docs/OpenSpec 或 inventory 来源

#### Scenario: recipe migration 候选
- **WHEN** config doctor 发现多个实体 YAML 可由同一 recipe 无损生成
- **THEN** doctor MUST 将其标记为 recipe migration candidate
- **AND** 只有在保留 experiment name、objective、dataset split、model/loss/training/output/checkpoint 语义和 focused tests 后，后续 change 才能删除实体 YAML

