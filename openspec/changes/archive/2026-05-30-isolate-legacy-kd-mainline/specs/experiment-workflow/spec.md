## ADDED Requirements

### Requirement: 默认实验入口去 KD-first 化
项目默认 quickstart、README 推荐入口、当前主线 quick validation 和新 canonical mainline 配置 MUST 以 no-KD supervised/adaptation 工作流为默认。KD 配置可以保留为 legacy 或 optional baseline，但不得作为当前主线默认实验入口。

#### Scenario: README quickstart 使用 no-KD 主线
- **WHEN** 开发者阅读 README 或当前主线运行说明
- **THEN** 推荐的首个训练、评估或 HiST-Beam LOSO 命令 MUST 使用 no-KD supervised/adaptation 配置
- **AND** 文档 MUST 不把 `logits_kd` 或 `rkd` 作为当前主线 quickstart

#### Scenario: canonical mainline 配置不要求 teacher checkpoint
- **WHEN** 用户加载当前推荐的 mainline 配置
- **THEN** 配置 MUST 能在没有 teacher checkpoint 的情况下完成解析和 dry-run/smoke 构建
- **AND** 输出 metadata MUST 记录 `distillation_enabled=false`

### Requirement: Legacy KD 配置生命周期可审计
保留的 KD 配置、脚本或虚拟 canonical recipe MUST 有明确生命周期分类。它们 MUST 被标注为 legacy KD、optional baseline、historical reproduction 或 future optional enhancement 中的一类，并且 MUST 不与 current mainline 配置混淆。

#### Scenario: KD 配置带生命周期标记
- **WHEN** 仓库中保留 `configs/**/logits_kd.yaml`、`configs/**/rkd.yaml` 或等价 KD recipe
- **THEN** 配置、相邻 README、inventory 或配置生成 metadata MUST 标明其生命周期分类
- **AND** 配置 MUST 记录它不是默认 mainline quick validation 的必要输入

#### Scenario: 未标记 KD 入口导致检查失败
- **WHEN** 开发者新增 KD 相关配置、script、tool 或 canonical recipe
- **THEN** 表面积或配置生命周期检查 MUST 要求登记该入口
- **AND** 未登记入口 MUST 导致检查失败或给出清晰修复提示

### Requirement: 实验 summary 区分 KD 与 mainline
训练、评估、LOSO 和 quick validation 的 summary artifact MUST 能区分 no-KD mainline、legacy KD baseline 和 optional KD enhancement。summary MUST 保留 KD 指标用于补充比较，但不得将其混入 mainline 默认排名或 eligibility 判断。

#### Scenario: summary 写出 method family
- **WHEN** 系统写出 run-level metrics、run metadata、LOSO summary 或 quick validation conclusion
- **THEN** artifact MUST 包含 method family、distillation enabled 状态或等价字段
- **AND** 对 legacy KD run MUST 记录 teacher checkpoint/source 和 distillation type

#### Scenario: mainline ranking 排除 legacy KD
- **WHEN** quick validation conclusion 计算 current mainline 排名或胜负判断
- **THEN** conclusion MUST 默认排除 `method_family=legacy_kd` 的 run
- **AND** 若用户显式请求 KD comparison，conclusion MUST 将其标记为 supplemental comparison

### Requirement: 项目描述反映当前主线
项目元数据、README 和高层文档 MUST 将当前项目主线描述为多模态/少样本跨场景 beam prediction、HiST-Beam 或 history-anchored adaptation，而不是 KD-first 工作流。历史 KD 背景可以保留，但必须标记为历史或 baseline。

#### Scenario: pyproject 描述不再 KD-first
- **WHEN** 开发者查看 `pyproject.toml` 的项目 description
- **THEN** description MUST 不把 knowledge distillation 描述为唯一或首要工作流
- **AND** 若提到 KD，MUST 表达其为 legacy 或 optional baseline

#### Scenario: 文档保留 KD 历史说明
- **WHEN** README 或 docs 提到历史 KD 代码
- **THEN** 文档 MUST 说明 KD 已从当前 active mainline 隔离
- **AND** 文档 MUST 指向 legacy/baseline 运行方式或历史 tag/branch 策略
