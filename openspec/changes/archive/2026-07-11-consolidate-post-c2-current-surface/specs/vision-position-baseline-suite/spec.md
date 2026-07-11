## REMOVED Requirements

### Requirement: Vision-Position baseline preset 矩阵
**Reason**: DeepSense6G image+GPS/GPS-only baseline suite 不再属于 post-C2 current surface。
**Migration**: 历史 preset 与结果保留在 archive；current baseline 以 AMBER/AMR 及受保护 MMW/CSI 路径为准。

#### Scenario: 旧 preset 不再构建
- **WHEN** current config loader 收到 vision-position preset
- **THEN** 它 MUST NOT 将该 preset 解析为 current baseline

### Requirement: 统一输入输出契约
**Reason**: 该 image+GPS baseline-specific shape 契约随 suite 退役。
**Migration**: current models 依据各自 owner spec 和 `ModelOutput` 契约维护 shape。

#### Scenario: current model 不依赖旧 baseline shape
- **WHEN** current training engine 处理 model output
- **THEN** 它 MUST NOT 需要 vision-position suite 的专用 shape adapter

### Requirement: 可配置的预处理、归一化和增强
**Reason**: 该 preprocessing matrix 只服务已退役 preset。
**Migration**: 受保护 dataset/model owner 继续管理自身的 preprocessing 和 normalization。

#### Scenario: 旧 preprocessing profile 不再要求
- **WHEN** current config 校验运行
- **THEN** 它 MUST NOT 要求 vision-position-specific image/GPS profile matrix

### Requirement: 训练评估闭环和 top-k 指标
**Reason**: 该要求重复通用 train/evaluate 契约，且专用 baseline 已退役。
**Migration**: current 训练和指标由 `training-evaluation-runtime` 与各 model owner 验证。

#### Scenario: 通用 workflow 继续独立工作
- **WHEN** current baseline 训练或评估
- **THEN** 它 MUST 使用通用 runtime，且 MUST NOT 依赖 vision-position suite

### Requirement: baseline metadata 和本地产物边界
**Reason**: 专用 preset metadata 字段没有 current producer 或 consumer。
**Migration**: 通用 provenance 和 ignored-output 边界由 current runtime/owner specs 维护。

#### Scenario: 旧 preset metadata 不再输出
- **WHEN** current run 写出 metadata
- **THEN** 它 MUST NOT 要求 `camera_ae_gps`、`resnet_gps` 或其他 vision-position preset 字段

### Requirement: baseline smoke 和回归测试
**Reason**: 只覆盖已退役 suite 的 tests 应与实现一并删除。
**Migration**: 保留 CLI help、config load 与 current model focused tests。

#### Scenario: 不保留退役 suite smoke
- **WHEN** consolidation 完成测试目录清理
- **THEN** tests MUST NOT 构建 vision-position-only presets

### Requirement: image+gps fusion baseline preset 选择
**Reason**: late-concat/transformer 二选一矩阵无 current config consumer。
**Migration**: 未来 fusion baseline 必须从新 change 和最小 preset 开始。

#### Scenario: fusion preset names 不再可用
- **WHEN** config 请求旧 image+gps late-concat 或 transformer preset
- **THEN** config validation MUST 拒绝该 current 路由

### Requirement: image+gps fusion encoder 可替换
**Reason**: Camera-AE/ResNet 可替换扩展面只属于已退役 suite。
**Migration**: 受保护 owner 只保留实际 current config 使用的 encoder 可配置性。

#### Scenario: 不保留推测性 encoder 切换
- **WHEN** current registry 构建模型
- **THEN** 它 MUST NOT 为 vision-position suite 保留 Camera-AE/ResNet 交换 facade
