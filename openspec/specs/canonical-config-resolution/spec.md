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

### Requirement: Snapshot canonical 配置解析
配置加载流程 MUST 能识别并生成 snapshot next-frame baseline 配置。可生成路径 MUST 包含单模态 `configs/<modality>/snapshot_next_frame_no_kd.yaml` 和 fusion `configs/fusion/<slug>_snapshot_next_frame_no_kd.yaml`。

#### Scenario: 生成单模态 snapshot 配置
- **WHEN** 用户加载缺失但合法的 `configs/gps/snapshot_next_frame_no_kd.yaml`
- **THEN** 系统 MUST 生成可用于训练和评估的 GPS snapshot 配置
- **AND** 最终配置 MUST 设置 `experiment.task: gps`
- **AND** 最终配置 MUST 设置 `data.dataset.seq_len: 1` 和 `data.dataset.num_pred: 1`
- **AND** 最终配置 MUST 设置 `data.dataset.train_csv_name: train_seqs_SNAPSHOT_NEXT_FRAME.csv`
- **AND** 最终配置 MUST 设置 `data.dataset.val_csv_name: val_seqs_SNAPSHOT_NEXT_FRAME.csv` 或等价 validation CSV 字段
- **AND** 最终配置 MUST 构建 `snapshot_frame` core

#### Scenario: 生成 fusion snapshot 配置
- **WHEN** 用户加载缺失但合法的 `configs/fusion/gps_mmwave_snapshot_next_frame_no_kd.yaml`
- **THEN** 系统 MUST 生成可用于训练和评估的 fusion snapshot 配置
- **AND** 最终配置 MUST 设置启用模态为 `["gps", "mmwave"]`
- **AND** 最终配置 MUST 设置 `experiment.task: fusion`
- **AND** 最终配置 MUST 构建无时序 snapshot fusion 模型

#### Scenario: 拒绝非法 snapshot slug
- **WHEN** 用户加载应被拒绝的非法路径 `configs/fusion/mmwave_gps_snapshot_next_frame_no_kd.yaml`
- **THEN** 系统 MUST 拒绝该路径
- **AND** 错误信息 MUST 提示 canonical slug 为 `gps_mmwave`

### Requirement: Snapshot 配置生成语义
生成的 snapshot 配置 MUST 明确覆盖历史窗口默认值，并保持实体 YAML 优先、命令行覆盖后应用和 schema 校验流程一致。

#### Scenario: 覆盖历史窗口默认值
- **WHEN** 系统生成任一 snapshot 配置
- **THEN** 生成配置 MUST 覆盖默认 `seq_len=8` 和 `num_pred=3`
- **AND** 生成配置 MUST 覆盖默认 GRU representation core
- **AND** 生成配置 MUST 覆盖默认历史窗口 CSV 为 snapshot 专用 train/validation CSV
- **AND** 生成配置 MUST 设置 `output.run_name` 包含 `snapshot_next_frame_no_kd`

#### Scenario: 实体 snapshot 配置优先
- **WHEN** 用户加载磁盘上存在的 snapshot YAML
- **THEN** 配置加载流程 MUST 使用实体 YAML 内容
- **AND** virtual snapshot 规则 MUST 不覆盖同名实体配置

#### Scenario: 命令行覆盖仍生效
- **WHEN** 用户加载 snapshot virtual 配置并传入覆盖项 `training.epochs=1`
- **THEN** 最终配置 MUST 使用 `training.epochs: 1`
- **AND** snapshot 必需字段 MUST 继续满足 `seq_len=1`、`num_pred=1` 和无时序 core 契约，除非用户显式退出 snapshot 变体并通过校验

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

### Requirement: 高级配置二次瘦身必须有候选分类
项目 MUST 在删除仍保留的高级实体 YAML 前维护候选分类。每个候选配置 MUST 被归入可由 recipe 无损生成、可由 recipe 生成但存在显式差异、或需要作为人工样例继续保留三类之一。

#### Scenario: 生成配置瘦身候选清单
- **WHEN** 开发者准备收敛 `configs/fusion/`、`configs/csi/hardening_matrix/` 或其它高级实验配置矩阵
- **THEN** 清单 MUST 记录每个候选实体 YAML 的分类、保留或删除理由和对应 recipe/overlay 名称
- **AND** 未分类的实体 YAML MUST 不得被删除

#### Scenario: 有差异的实体配置先记录差异
- **WHEN** 某个实体 YAML 与候选 recipe 在模型、loss、training schedule、dataset 字段或 checkpoint 来源上存在差异
- **THEN** 该差异 MUST 先记录为允许差异、overlay option 或保留理由
- **AND** 不得把该实体 YAML 当作无损可生成配置直接删除

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

### Requirement: Difficulty 配置解析与校验
配置加载流程 MUST 支持解析 top-level 或等价位置的 difficulty profiles，并在实体 YAML、virtual canonical 配置和命令行覆盖之后执行标准化与校验。解析结果 MUST 包含 profile id、operator list、stage/split selector、condition、severity、seed、affected modalities、fallback 和 digest。未知 operator、未知模态、非法 stage/split、非法 severity 或 target-shift 配置 MUST 被拒绝。

#### Scenario: 实体配置解析 difficulty profiles
- **WHEN** 用户加载包含 difficulty profiles 的实体 YAML
- **THEN** 配置加载流程 MUST 在 defaults、overlay 和命令行覆盖后标准化 difficulty 配置
- **AND** `final_config.yaml` 或 resolved config MUST 记录标准化后的 profile 和 digest

#### Scenario: 命令行覆盖 difficulty severity
- **WHEN** 用户通过命令行覆盖某个 difficulty profile 的 severity
- **THEN** 覆盖 MUST 在 difficulty validation 前生效
- **AND** resolved profile digest MUST 反映覆盖后的参数

#### Scenario: 非法 stage 被拒绝
- **WHEN** 配置声明 difficulty stage `preprocess_dataset_files`
- **THEN** 系统 MUST 拒绝配置
- **AND** 错误信息 MUST 列出允许的 stage，例如 train、validation、test、evaluation 和 benchmark

### Requirement: Difficulty overlay recipe
canonical/virtual config recipe MAY 生成当前支持的 difficulty overlay，例如 clean、GPS mild async、GPS severe async、GPS/image dropout training、image hard degradation 和 benchmark sweep。recipe 生成的 difficulty 配置 MUST 与实体 YAML 使用同一标准化、validation 和 digest 流程。已退役的旧 KD、G2D、CRAF、MARF 或 image motion profile MUST 不得通过 difficulty overlay 恢复。

#### Scenario: 生成 mild async training overlay
- **WHEN** 用户加载声明支持的 mild async training overlay 配置
- **THEN** 系统 MUST 生成 train-stage GPS async difficulty profile
- **AND** 其它 supervised/fusion 配置语义 MUST 继续由当前 canonical recipe 决定

#### Scenario: difficulty overlay 不恢复 image motion profile
- **WHEN** 用户加载 image degradation difficulty overlay
- **THEN** image modality input profile MUST 仍解析为当前 RGB/ImageNet profile
- **AND** 系统 MUST 不生成或接受已删除的 `motion_mask` image profile、image motion cache 或 motion encoder

### Requirement: Difficulty 解析产物可比较
系统 MUST 为 difficulty profiles 生成稳定 digest，用于 run metadata、benchmark comparability 和论文图表分组。digest MUST 基于标准化后的 operator、condition、severity、stage/split、seed 和 fallback，而不是用户 YAML 字段顺序。

#### Scenario: 字段顺序不同 digest 相同
- **WHEN** 两个配置声明语义相同但 YAML 字段顺序不同的 difficulty profile
- **THEN** 标准化后的 digest MUST 相同
- **AND** benchmark comparability MAY 将它们视为同一 difficulty condition

#### Scenario: severity 改变 digest 改变
- **WHEN** 两个 profile 仅 severity 不同
- **THEN** digest MUST 不同
- **AND** 输出指标 MUST 能按不同 severity 分组
