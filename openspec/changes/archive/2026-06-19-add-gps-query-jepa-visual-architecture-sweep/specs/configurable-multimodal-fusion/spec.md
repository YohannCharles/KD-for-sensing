# configurable-multimodal-fusion Specification Delta

## ADDED Requirements

### Requirement: Architecture sweep 派生配置
项目 MUST 提供 GPS-query JEPA visual architecture sweep 派生配置或配置生成规则。每个派生配置 MUST 继承匹配 baseline 的数据 split、Image+GPS 模态、beam objective、label space、metric profile、GPS feature mode、训练 recipe 关键字段和输出边界，只覆盖 architecture sweep 变量。

#### Scenario: 派生配置只覆盖架构变量
- **WHEN** 开发者加载 architecture sweep 派生配置
- **THEN** 配置 MUST 使用现有 `modular_sequence` 或已登记 component baseline 路径
- **AND** 配置 MUST 只显式覆盖 visual encoder、pooler、adapter、representation core、freeze policy、parameter groups、run name 或 ablation metadata

#### Scenario: baseline 配置不被替换
- **WHEN** 新增 architecture sweep 配置族
- **THEN** 系统 MUST 不删除、重命名或语义替换现有 Image ResNet+GPS、JEPA GPS-biased mean-pooling、JEPA GPS-query 或 Predictive GPS-query++ baseline 配置
- **AND** README 或实验说明 MUST 指出 sweep 候选应与匹配 baseline 成对比较

### Requirement: Architecture sweep 配置可加载性
architecture sweep 中的每个实体 YAML、virtual config 或生成配置 MUST 能通过项目配置加载器加载并构建模型 smoke。配置加载 MUST 不要求本地 checkpoint 存在，除非该测试显式选择 checkpoint strict loading。

#### Scenario: sweep 配置加载 smoke
- **WHEN** focused config test 遍历 architecture sweep smoke 配置
- **THEN** 每个配置 MUST 解析成功并暴露 model、data、training、evaluation 和 output 基本字段
- **AND** 配置 metadata MUST 包含 `variant_id`、`family`、`checkpoint_policy` 和 strict comparability fields 或其继承来源

#### Scenario: checkpoint path 缺失时可诊断
- **WHEN** 配置引用的 checkpoint path 在本地不存在
- **THEN** 普通配置加载测试 MUST 不因缺失 checkpoint 失败
- **AND** 需要实际加载权重的 forward test MUST 抛出包含 checkpoint path 和 variant id 的清晰错误或使用 mock checkpoint

### Requirement: Architecture sweep 不新增旧入口
architecture sweep MUST 不新增 root-level 旧式训练脚本、兼容聚合层、退役研究线实体配置或绕过 `src/kd_sensing` 包结构的运行方式。运行命令 MUST 复用现有 `scripts/train.py`、`scripts/evaluate.py`、包内 CLI 或已登记薄 alias。

#### Scenario: 运行命令使用当前入口
- **WHEN** sweep manifest 写出 train/evaluate command
- **THEN** command MUST 使用当前允许的训练、评估或诊断入口
- **AND** Python 相关命令 MUST 使用 `conda run -n kd_mm_beam`

#### Scenario: 退役路线不回流
- **WHEN** 新增 sweep 配置或文档
- **THEN** 系统 MUST 不恢复旧 KD、HiST/Hist、Top8 selector、camera residual、GPS residual、G2D/CRAF/MARF 或 root-level legacy script 路线
- **AND** 架构边界测试 MUST 能覆盖至少一个防回流检查或配置 allowlist 检查
