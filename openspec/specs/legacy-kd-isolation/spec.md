# legacy-kd-isolation Specification

## Purpose
定义历史 teacher-student KD 入口退役后的拒绝、历史读取和 summary 隔离边界。

## Requirements

### Requirement: Legacy KD 入口必须拒绝
系统 MUST 不再运行 teacher-student KD。旧 `logits_kd`、`rkd`、`distillation.*`、`teacher_model_name` 或旧 `*_no_kd` 路径只允许作为 migration guard、历史 artifact 读取或 archive 说明中的命中。

#### Scenario: 旧 KD 配置被拒绝
- **WHEN** 用户请求旧 `logits_kd`、`rkd` 或等价 KD 配置
- **THEN** 配置加载 MUST 失败
- **AND** 错误信息 MUST 指向 strong、lightweight、supervised 或 adaptation 入口

#### Scenario: 旧 distillation override 被拒绝
- **WHEN** 用户通过命令行传入 `distillation.*`、`teacher_model_name` 或 `kd_mode`
- **THEN** 配置加载 MUST 失败
- **AND** 系统 MUST 不构建 frozen teacher 或 distiller

### Requirement: 历史 KD metadata 只读隔离
历史 run metadata MAY 继续包含 `distillation_enabled` 或 `method_family=legacy_kd`，但新训练产物 MUST 不写出这些字段作为当前运行模式。读取历史 summary 时，系统 MUST 将这些记录标记为历史或 supplemental，不得纳入当前主结论排名。

#### Scenario: 历史 KD run 被识别
- **WHEN** summary 读取一个历史 `method_family=legacy_kd` 或 `distillation_enabled=true` 的 run
- **THEN** 系统 MAY 展示该 run 作为历史资料
- **AND** 系统 MUST 不把该 run 计入当前 mainline ranking

#### Scenario: 新训练产物不写 KD lineage
- **WHEN** 当前 supervised/adaptation 训练完成
- **THEN** run metadata MUST 不新增 `method_family=legacy_kd`
- **AND** run metadata MUST 不新增 `distillation_enabled=true`
### Requirement: Legacy KD 已删除
系统 MUST 不再保留 legacy KD 代码、配置、测试或运行时入口。`logits_kd`、`rkd`、teacher-student KD、KD baseline summary 和 KD virtual alias MUST 被视为已删除能力。

#### Scenario: 显式 legacy KD 被拒绝
- **WHEN** 用户请求运行 `logits_kd`、`rkd` 或等价 legacy KD baseline
- **THEN** 系统 MUST 拒绝该请求
- **AND** 错误信息 MUST 指向当前 supervised/adaptation workflow

#### Scenario: 新方法不得复用 legacy KD runtime
- **WHEN** 开发者新增 HiST-Beam residual、prototype、calibration 或其它方法
- **THEN** 实现 MUST 不依赖 legacy KD teacher-student forward 逻辑
- **AND** 仓库 MUST 不包含可复用的 legacy KD runtime 聚合入口

### Requirement: Fusion teacher 支持 GPS
`fusion_teacher` MUST 能在启用 GPS 时融合 GPS 特征，并保持输出契约 `(pred, input_features, output_features)`。GPS 分支 MUST 使用与 GPS-only teacher 兼容的 feature extraction 风格。

#### Scenario: fusion_teacher 使用 GPS
- **WHEN** `fusion_teacher` 配置包含 `gps`
- **THEN** 模型 MUST 接收 GPS-Rel-Polar 输入张量 `[B, T, 3]`
- **AND** 模型 MUST 将 GPS 分支 embedding 与其它启用模态 embedding 在 feature 维拼接
- **AND** 模型 MUST 返回 `[B, T, num_classes]` logits

#### Scenario: fusion_teacher 缺少启用模态输入
- **WHEN** `fusion_teacher` 配置包含 `gps` 但 forward 未收到 GPS 输入
- **THEN** 系统 MUST 抛出清晰错误

### Requirement: Fusion student 支持 GPS
`fusion_student` MUST 能在启用 GPS 时融合 GPS 特征，并保持 lightweight student 语义。GPS student 分支 MUST 使用轻量 MLP 或投影层，且默认 output hidden size MUST 与 teacher 对齐以支持 RKD。

#### Scenario: fusion_student 使用 GPS
- **WHEN** `fusion_student` 配置包含 `gps`
- **THEN** 模型 MUST 接收 GPS-Rel-Polar 输入张量 `[B, T, 3]`
- **AND** 模型 MUST 将 GPS 分支 embedding 与其它启用模态 embedding 在 feature 维拼接
- **AND** 模型 MUST 返回 `[B, T, num_classes]` logits

#### Scenario: fusion_student KD 兼容
- **WHEN** fusion KD 配置中的 teacher 和 student 使用相同的 `modalities`
- **THEN** 系统 MUST 能完成 teacher/student forward
- **AND** logits KD 与 RKD MUST 能接收 fusion teacher/student 的 logits、input_features 和 output_features

### Requirement: Fusion teacher 支持 LiDAR
`fusion_teacher` MUST 能在启用 LiDAR 时融合 LiDAR BEV 特征，并保持输出契约 `(pred, input_features, output_features)`。LiDAR 分支 MUST 使用与 LiDAR-only teacher 兼容的 feature extraction 风格。

#### Scenario: fusion_teacher 使用 LiDAR
- **WHEN** `fusion_teacher` 配置包含 `lidar`
- **THEN** 模型 MUST 接收 LiDAR BEV 输入张量 `[B, T, C, H, W]`
- **AND** 模型 MUST 将 LiDAR 分支 embedding 与其它启用模态 embedding 在 feature 维拼接
- **AND** 模型 MUST 返回 `[B, T, num_classes]` logits

#### Scenario: fusion_teacher 缺少 LiDAR 输入
- **WHEN** `fusion_teacher` 配置包含 `lidar` 但 forward 未收到 LiDAR 输入
- **THEN** 系统 MUST 抛出清晰错误

### Requirement: Fusion student 支持 LiDAR
`fusion_student` MUST 能在启用 LiDAR 时融合 LiDAR BEV 特征，并保持 lightweight student 语义。LiDAR student 分支 MUST 使用轻量 CNN 或 depthwise separable convolution，并通过 adaptive pooling 生成固定长度帧级 embedding。

#### Scenario: fusion_student 使用 LiDAR
- **WHEN** `fusion_student` 配置包含 `lidar`
- **THEN** 模型 MUST 接收 LiDAR BEV 输入张量 `[B, T, C, H, W]`
- **AND** 模型 MUST 将 LiDAR 分支 embedding 与其它启用模态 embedding 在 feature 维拼接
- **AND** 模型 MUST 返回 `[B, T, num_classes]` logits

#### Scenario: fusion_student LiDAR KD 兼容
- **WHEN** fusion 配置中的 primary model 使用包含 LiDAR 的 `modalities`
- **THEN** 系统 MUST 能完成 primary model forward
- **AND** loss MUST 能接收 fusion primary model 的 logits、input_features 和 output_features

### Requirement: Fusion teacher 支持 mmWave
`fusion_teacher` MUST 能在启用 mmWave 时融合 mmWave 64 维 receive-power 特征，并保持输出契约 `(pred, input_features, output_features)`。mmWave 分支 MUST 使用与 mmWave-only teacher 兼容的 feature extraction 风格。

#### Scenario: fusion_teacher 使用 mmWave
- **WHEN** `fusion_teacher` 配置包含 `mmwave`
- **THEN** 模型 MUST 接收 mmWave 输入张量 `[B, T, 64]`
- **AND** 模型 MUST 将 mmWave 分支 embedding 与其它启用模态 embedding 在 feature 维拼接
- **AND** 模型 MUST 返回 `[B, T, num_classes]` logits

#### Scenario: fusion_teacher 缺少 mmWave 输入
- **WHEN** `fusion_teacher` 配置包含 `mmwave` 但 forward 未收到 mmWave 输入
- **THEN** 系统 MUST 抛出清晰错误

### Requirement: Fusion student 支持 mmWave
`fusion_student` MUST 能在启用 mmWave 时融合 mmWave 64 维 receive-power 特征，并保持 lightweight student 语义。mmWave student 分支 MUST 使用轻量 MLP 或投影层，且默认 output hidden size MUST 与 teacher 对齐以支持 RKD。

#### Scenario: fusion_student 使用 mmWave
- **WHEN** `fusion_student` 配置包含 `mmwave`
- **THEN** 模型 MUST 接收 mmWave 输入张量 `[B, T, 64]`
- **AND** 模型 MUST 将 mmWave 分支 embedding 与其它启用模态 embedding 在 feature 维拼接
- **AND** 模型 MUST 返回 `[B, T, num_classes]` logits

#### Scenario: fusion_student mmWave KD 兼容
- **WHEN** fusion KD 配置中的 teacher 和 student 使用包含 mmWave 的相同 `modalities`
- **THEN** 系统 MUST 能完成 teacher/student forward
- **AND** logits KD 与 RKD MUST 能接收 fusion teacher/student 的 logits、input_features 和 output_features

### Requirement: Fusion teacher image 分支复用单模态特征提取器
`fusion_teacher` 在启用 image 模态时 MUST 使用 image-only teacher 暴露的 `ImageFeatureExtractor` 作为帧级特征提取器。系统 MUST 不再为 fusion teacher 维护单独的旧版 image feature extractor 副本。

#### Scenario: 构建包含 image 的 fusion teacher
- **WHEN** 用户构建 `fusion_teacher` 且 `modalities` 包含 `image`
- **THEN** 模型 MUST 将 `image_feature_extractor` 初始化为 `ImageFeatureExtractor`
- **AND** image 分支输出 MUST 保持 `[B, T, feature_size]` 形状以参与 fusion projection

#### Scenario: 构建不包含 image 的 fusion teacher
- **WHEN** 用户构建 `fusion_teacher` 且 `modalities` 不包含 `image`
- **THEN** 模型 MUST 不创建 image feature extractor
- **AND** 缺失 image 输入不得阻止该 fusion teacher forward

#### Scenario: 旧 fusion teacher checkpoint 结构不匹配
- **WHEN** 用户使用严格加载将旧 `FusionImageFeatureExtractor` 结构的 `fusion_teacher` checkpoint 加载到新模型
- **THEN** 系统 MUST 报告 checkpoint 结构不匹配
- **AND** 错误信息 MUST 包含 missing keys 或 unexpected keys 诊断

### Requirement: Fusion 模型公开类名表达 teacher/student 职责
Early-concat fusion teacher 和 student MUST 暴露职责明确的公开 Python 类名。`fusion_teacher` 注册名 MUST 构建 `FusionTeacherModalityNet`，`fusion_student` 注册名 MUST 构建 `FusionStudentModalityNet`。旧类名 `old fusion teacher class alias` 和 `old fusion student class alias` MUST 不再作为兼容 alias 导出。

#### Scenario: 构建 fusion teacher 返回新类名
- **WHEN** 开发者通过 `MODELS.build()` 构建 `type: fusion_teacher`
- **THEN** 系统 MUST 返回 `FusionTeacherModalityNet` 实例
- **AND** 该实例 MUST 保持既有 `fusion_teacher` forward 输出契约

#### Scenario: 构建 fusion student 返回新类名
- **WHEN** 开发者通过 `MODELS.build()` 构建 `type: fusion_student`
- **THEN** 系统 MUST 返回 `FusionStudentModalityNet` 实例
- **AND** 该实例 MUST 保持既有 `fusion_student` forward 输出契约

#### Scenario: 旧类名 alias 被拒绝
- **WHEN** 开发者导入 `old fusion teacher class alias` 或 `old fusion student class alias`
- **THEN** 导入 MUST 失败或触发清晰迁移错误
- **AND** 错误信息 MUST 指向 `FusionTeacherModalityNet` 或 `FusionStudentModalityNet`

### Requirement: 已退役 KD baseline 不影响 canonical 模态 slug 解析
删除 fusion KD virtual modes 后，canonical 模态 slug 解析 MUST 继续支持当前合法模态集合、顺序规范化、重复模态拒绝、未知模态拒绝和单模态转发建议。

#### Scenario: canonical slug 校验保持稳定
- **WHEN** 用户请求当前 fusion virtual config，并使用合法模态集合
- **THEN** 系统 MUST 按固定模态顺序解析 slug 并生成配置
- **AND** 重复模态、未知模态或可转为单模态配置的路径 MUST 继续给出清晰错误或建议

### Requirement: Legacy fusion whole-model routes are retired
普通 fusion baseline MUST 优先使用 `modular_sequence` 组件化路径。旧 `fusion_lightweight` 和无 current config 依赖的 `fusion_strong` whole-model 注册名 MUST 被 removed guard 拒绝；保留的 fusion whole-model 注册名必须有 current spec 或 whole-model exception 理由。

#### Scenario: radar+GPS supervised fusion 使用 modular_sequence
- **WHEN** 用户加载 `configs/fusion/radar_gps_supervised.yaml`
- **THEN** 最终配置的 `model.primary.type` MUST 为 `modular_sequence`
- **AND** 配置 MUST 使用 `radar_cnn`、`gps_mlp`、projectors、`early_concat_gru` 或等价 current representation core
- **AND** fusion task runtime MUST 继续只准备启用模态的 batch 输入

#### Scenario: 请求 legacy fusion 注册名
- **WHEN** 用户请求 `fusion_lightweight` 或 `fusion_strong`
- **THEN** registry MUST 抛出 removed component 错误
- **AND** 错误信息 MUST 建议使用 `modular_sequence` fusion 配置

#### Scenario: current fusion whole-model exceptions 不受影响
- **WHEN** 用户配置 current 保留的 `cls_token_transformer_fusion` 或 `token_transformer_fusion`
- **THEN** 系统 MUST 继续按对应 current spec 或 config 构建模型
- **AND** 本 change MUST 不改变这些保留模型的 forward/output 契约

### Requirement: Legacy model registry names are retired with migration guards
项目 MUST 将已退役的 legacy model、encoder、core 和 head 注册名排除在 current 可构建组件之外。对仍有当前迁移价值的旧名称，项目 MAY 保留 removed guard 并给出明确迁移目标；对完全退役且不再承诺兼容的旧名称，项目 MAY 删除 guard table 并让 registry 使用普通 unknown-name 错误。

#### Scenario: 旧整模型注册名被拒绝
- **WHEN** 用户通过 `MODELS.build()` 请求 `radar_strong`、`gps_lightweight`、`mmwave_strong`、`fusion_lightweight` 或其它本 change 退役的旧整模型注册名
- **THEN** 系统 MUST 拒绝构建该名称
- **AND** 若 removed guard 被保留，错误信息 MUST 包含请求名称、registry 名称和 `modular_sequence` 迁移目标；若 guard 已删除，系统 MAY 使用普通 unknown-name 错误

#### Scenario: 旧别名被拒绝
- **WHEN** 用户请求 `modular_sequence_model`、`gps_only_neural_baseline`、`jepa_token_transformer` 或 `safe_residual_reranker`
- **THEN** 系统 MUST 不把这些名称注册为 current 可构建组件
- **AND** 若别名仍在 current migration table 中，错误信息 MUST 指向对应 canonical 名称或配置路径

#### Scenario: feature extractor 不作为完整模型列出
- **WHEN** 默认组件导入完成后开发者查看 `MODELS.list()`
- **THEN** 输出 MUST NOT 包含 `radar_feature_extractor`、`lidar_feature_extractor` 或 `mmwave_feature_extractor`
- **AND** 对应 feature extractor 类 MAY 继续通过窄模块导入或由 encoder 组件内部复用

#### Scenario: current registry discovery 只列当前入口
- **WHEN** 文档、架构摘要或架构边界测试检查 current registry surface
- **THEN** current model/encoder/core/head 清单 MUST 不把 removed guard 名称展示为可推荐入口
- **AND** removed 名称 MAY 出现在退役边界、migration table 或普通错误路径中

### Requirement: Removed guard 表只保留当前迁移价值
注册表和配置 guard MAY 为仍常见或仍有当前迁移路径的旧名称提供专属 removed error。完全退役且已由 OpenSpec tombstone、inventory 和 README retired wording 覆盖的历史路线 MUST 不要求每个 registry 或 facade 继续维护专属 removed guard。

#### Scenario: 保留高频迁移 guard
- **WHEN** 用户请求 `scenario31` dataset alias、KD loss token、removed image profile 或 removed image encoder
- **THEN** 系统 SHOULD 继续给出清晰迁移错误
- **AND** 错误 MUST 指向当前 canonical dataset、loss 或 image profile 入口

#### Scenario: 低价值 retired 名称回落 unknown
- **WHEN** 用户请求完全退役且不再有当前迁移目标的旧研究线 registry 名称
- **THEN** 系统 MAY 返回普通 unknown-name registry 错误
- **AND** 系统 MUST 不通过 deprecated alias、facade 或 virtual config 重定向到当前实现

### Requirement: Removed guard 只保留有迁移价值的名称
`register_removed()` 或等价 removed-name guard MUST 只用于仍可能从当前迁移路径触发、且普通 unknown-name 错误不足以防止误用的名称。完全退役、已有 OpenSpec tombstone 或只由测试 fixture 引用的名称 MUST 回落为 unknown-name，除非设计说明记录保留理由。

#### Scenario: 低价值 removed-name guard 被删除
- **WHEN** 某个 removed-name guard 只服务历史 fixture 或已退役研究线文案
- **THEN** 本 change MAY 删除该 guard
- **AND** 对应测试 MUST 改为验证 current registry 不注册该名称，而不是要求专属迁移文案

### Requirement: 已删除组件错误可诊断
当用户引用已删除的兼容组件名称或退役研究线组件名称时，注册表错误 MUST 至少包含请求名称、registry 名称或可用名称上下文。对于仍有当前迁移价值的名称，错误 MUST 区分“未知名称”和“已删除名称”并给出迁移方向；对于完全退役且不再承诺兼容的历史名称，系统 MUST 允许使用普通 unknown-name 错误、配置 migration guard 或集中退役说明替代长期 removed guard table。registry 实现 MUST 不为了历史说明长期维护没有 current migration value 的 removed-name 表项。

#### Scenario: 已删除 dataset type
- **WHEN** 用户请求构建 `scenario9` dataset 且项目仍保留该迁移说明
- **THEN** 系统 MUST 抛出包含 `scenario9` 的错误
- **AND** 错误信息 MUST 说明该名称已删除并给出 `deepsense6g + scene` 配置示例

#### Scenario: 已删除模型 alias
- **WHEN** 用户请求旧 fusion 类名 alias 或已删除 image encoder alias，且该名称仍在 current migration table 中
- **THEN** 系统 MUST 抛出包含请求名称的错误
- **AND** 错误信息 MUST 列出当前支持的 canonical 注册名

#### Scenario: 已退役研究线组件
- **WHEN** 用户请求 `craf_fusion`、`marf_fusion`、`g2d` distiller 或 `multimodal_nf` dataset
- **THEN** 系统 MUST 拒绝构建
- **AND** 系统 MUST 不通过 deprecated alias、overlay 或兼容 facade 重定向到其它实现

#### Scenario: 完全退役名称不要求 removed table
- **WHEN** 某个历史组件名称已经由 retired-tombstone spec、配置 migration guard 或文档生命周期边界覆盖，且没有当前迁移路径
- **THEN** registry MUST 允许不在 removed-name table 中保留该名称
- **AND** unknown-name 错误 MUST 仍列出 registry 名称、请求名称或可用 canonical 名称上下文

### Requirement: Radar-only KD 实验配置已移除
项目 MUST 不再提供 radar-only KD 配置。旧 `logits_kd` 和 `rkd` 配置 MUST 在配置解析阶段失败，并引导用户使用 `configs/radar/strong.yaml`、`configs/radar/lightweight.yaml` 或 `configs/radar/supervised.yaml`。

#### Scenario: 使用 logits KD 启动 radar-only 训练
- **WHEN** 用户通过训练入口传入 `configs/radar/logits_kd.yaml`
- **THEN** 系统 MUST 拒绝该配置
- **AND** 系统 MUST 不构建 frozen teacher 或 distiller

#### Scenario: 使用 RKD 启动 radar-only 训练
- **WHEN** 用户通过训练入口传入 `configs/radar/rkd.yaml`
- **THEN** 系统 MUST 拒绝该配置
- **AND** 系统 MUST 不构建 frozen teacher 或 distiller

#### Scenario: 旧 RadarTeacher checkpoint 自动解析被移除
- **WHEN** 用户运行当前 radar 训练配置
- **THEN** 系统 MUST 不解析 teacher checkpoint
- **AND** 训练流程 MUST 只更新 `model.primary`

#### Scenario: 旧 RadarTeacher checkpoint override 被拒绝
- **WHEN** 用户通过命令行覆盖 `distillation.teacher_model_name`
- **THEN** 配置加载 MUST 失败
- **AND** 错误信息 MUST 指向当前 supervised/adaptation 入口

### Requirement: RadarStudent legacy no-KD 请求迁移
项目 MUST 拒绝旧 `configs/radar/student_no_kd.yaml` 入口，并将其解释为历史 no-KD/student 路径的 migration guard。当前 radar 轻量或 supervised 实验 MUST 使用 `configs/radar/lightweight.yaml`、`configs/radar/supervised.yaml` 或等价 `model.primary` 配置。

#### Scenario: 旧 RadarStudent no-KD 请求迁移
- **WHEN** 用户通过训练入口传入已退役的 `configs/radar/student_no_kd.yaml`
- **THEN** 系统 MUST 拒绝该旧入口
- **AND** 错误信息 MUST 指向当前 radar lightweight 或 supervised 配置

### Requirement: 单模态 legacy no-KD 入口兼容
项目 MUST 拒绝现有 `configs/<modality>/no_kd.yaml` 旧入口，并 MUST 在文档中说明其历史语义和推荐替代入口。

#### Scenario: image legacy no-KD 保持 student baseline
- **WHEN** 用户运行已退役的 `configs/image/no_kd.yaml`
- **THEN** 系统 MUST 拒绝该配置
- **AND** 文档 MUST 引导新实验优先使用 `configs/image/lightweight.yaml` 或 `configs/image/supervised.yaml`

#### Scenario: radar GPS LiDAR legacy no-KD 保持 teacher baseline
- **WHEN** 用户运行已退役的 `configs/radar/no_kd.yaml`、`configs/gps/no_kd.yaml` 或 `configs/lidar/no_kd.yaml`
- **THEN** 系统 MUST 拒绝这些配置
- **AND** 文档 MUST 引导新实验优先使用对应 `strong.yaml`、`lightweight.yaml` 或 `supervised.yaml`

### Requirement: primary 角色不得受原脚本残留影响
配置驱动流程 MUST 以 YAML 中的 `model.primary` 作为被训练主模型。默认 canonical lightweight baseline MUST 使用 lightweight 注册名，不得默认使用旧 teacher-as-student 残留。

#### Scenario: no-KD 只训练配置中的主模型
- **WHEN** 配置使用当前 supervised/adaptation 入口
- **THEN** 训练流程 MUST 不构建或加载 frozen teacher
- **AND** optimizer MUST 只更新 `model.primary` 构建出的主模型

#### Scenario: canonical student baseline 使用 lightweight student
- **WHEN** 开发者加载任意 canonical `lightweight.yaml`
- **THEN** `model.primary.type` MUST 为对应 lightweight 注册名
- **AND** `model.primary.type` MUST NOT 等于对应 strong 注册名

#### Scenario: canonical KD 路径被拒绝
- **WHEN** 开发者加载任意 canonical `logits_kd.yaml` 或 `rkd.yaml`
- **THEN** 配置加载 MUST 失败
- **AND** 系统 MUST 不构建 frozen teacher 或 distiller
