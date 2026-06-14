## ADDED Requirements

### Requirement: 主线实验文档索引检查
项目健康护栏 MUST 检查主线实验文档治理所需的 current 文档被创建、索引并登记生命周期。检查 MUST 不读取真实 `dataset/`、`outputs/`、checkpoint、cache、metrics 或 logs。

#### Scenario: 主线文档缺失或未索引
- **WHEN** 架构边界或文档健康检查运行
- **THEN** 检查 MUST 验证 README 或文档索引能指向主线模型目录、实验协议表和结果/claim 账本
- **AND** 缺失链接时检查 MUST 失败或给出明确修复信息

#### Scenario: 新增 capability 未登记 lifecycle
- **WHEN** `openspec/specs/mainline-experiment-documentation/spec.md` 存在
- **THEN** `docs/project_surface_inventory.md` 的 OpenSpec capability lifecycle 分类 MUST 将 `mainline-experiment-documentation` 标记为 current
- **AND** 未分类或分类为 retired/supporting 时检查 MUST 失败

### Requirement: current 文档结果状态 wording 检查
项目健康护栏 MUST 检查 current docs 和 current specs 不得把 mock、smoke、debug、lowmem、upper-bound、historical ablation、local substitute 或 blocked official reproduction 写成无 caveat 的正式结果。检查 MAY 使用关键词和限定词的静态规则，但 MUST 允许明确标记的历史或 appendix 段落。

#### Scenario: upper-bound 缺少限定
- **WHEN** current 文档中出现 `test_as_validation`、`upper-bound` 或等价用 test split 选 checkpoint 的说明
- **THEN** 附近文本 MUST 标明 upper-bound、非 official、不可作为 strict/unseen evaluation 或仅用于上限诊断
- **AND** 缺少限定时健康检查 MUST 失败

#### Scenario: future target 缺少历史限定
- **WHEN** current 文档中出现 `target-beam-source future`、`target_beam_source: future` 或等价 future target Table III 说明
- **THEN** 附近文本 MUST 标明 historical ablation、sequence-prediction ablation 或不得作为 Table III strict setup
- **AND** 缺少限定时健康检查 MUST 失败

#### Scenario: mock 或 smoke 数值缺少 caveat
- **WHEN** current 文档中出现 mock、dry-run、smoke、synthetic 或极小样本指标
- **THEN** 附近文本 MUST 说明该数值只验证代码路径或 schema，不用于论文/正式结果比较
- **AND** 缺少 caveat 时健康检查 MUST 失败

### Requirement: current spec 内部旧 active wording 检查
项目健康护栏 MUST 检查 current specs 内部不得同时保留旧 active workflow wording 与当前退役/拒绝 wording。已退役路线至少包括 legacy KD、teacher/student KD runtime、`teacher_no_kd`、`student_no_kd`、`logits_kd`、`rkd`、HiST/Hist、Raymobtime s008、standalone Top8 selector、GPS residual 和 camera residual。

#### Scenario: experiment-workflow 不保留 KD active 构建要求
- **WHEN** 健康检查扫描 `openspec/specs/experiment-workflow/spec.md`
- **THEN** 未加退役、拒绝、历史或 migration guard 限定的 `KD/loss`、`kd_mode`、teacher/student 成对训练、`student_no_kd` 当前入口 wording MUST 被视为漂移
- **AND** 检查 MUST 要求通过 OpenSpec change 清理为 current `model.primary` 语义或 retired/supporting 语义

#### Scenario: supporting helper 不被误判为 current workflow
- **WHEN** current spec 提到 TopK、LOSO、artifact registry、metric helper 或 migration guard
- **THEN** 文档 MUST 指明其 supporting、helper、guard 或被当前 workflow 消费的边界
- **AND** 文档 MUST 不把旧 standalone workflow 恢复为当前推荐入口

### Requirement: 文档健康检查无运行副作用
主线文档和规格漂移检查 MUST 只读取已跟踪源码、配置、文档、OpenSpec artifact 和测试文件。检查 MUST 不启动真实训练、不读取真实数据、不扫描 ignored 运行产物、不写入 checkpoint 或结果。

#### Scenario: 检查不读取本地产物
- **WHEN** 开发者运行文档健康检查或架构边界测试
- **THEN** 检查 MUST 不打开 `dataset/` 真实文件、`outputs/` metrics、checkpoint、feature cache 或 TensorBoard event
- **AND** 检查 MUST 只基于已跟踪文档、配置和 OpenSpec artifact 判断文档漂移

#### Scenario: Python 检查使用项目环境
- **WHEN** 文档健康检查通过 Python 测试实现
- **THEN** 推荐命令 MUST 使用 `conda run -n kd_mm_beam pytest ...`
- **AND** 该测试 MUST 不要求真实 GPU、真实数据或训练产物可用
