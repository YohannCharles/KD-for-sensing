## ADDED Requirements

### Requirement: 研究运行 happy path
项目 MUST 提供或记录一个当前研究 happy path，用于在不启动真实训练的情况下汇总当前状态、运行轻量健康检查、生成 evidence checklist 或静态预览入口。该 happy path MUST 默认不读取真实 `dataset/`、不加载 checkpoint、不写入训练产物、不移动或删除本地产物。

#### Scenario: 默认 happy path 无训练副作用
- **WHEN** 开发者运行默认 research run preview 命令
- **THEN** 命令 MUST 只执行 OpenSpec/架构/doctor/run-index/dashboard/table-consistency 或等价无副作用步骤
- **AND** 命令 MUST 不调用长时间 `kd-sensing-train` 真实训练
- **AND** 输出 MUST 写入 ignored output root、pytest 临时目录或用户显式路径

#### Scenario: 真实训练必须显式 opt-in
- **WHEN** 用户需要启动真实训练、fresh eval、checkpoint 写入或 cache rebuild
- **THEN** 命令 MUST 要求显式参数、配置或 manifest
- **AND** 文档 MUST 说明预计产物仍属于 ignored runtime artifacts，不得自动纳入源码变更

### Requirement: 静态 evidence preview QA
项目 MUST 为 HTML、CSV、figure data、paper table、checklist 和 conclusion draft 提供静态 evidence QA 或等价 focused validation。QA MUST 使用 synthetic fixture 或用户显式输入，不得依赖真实 dataset、真实 checkpoint 或远程网络。

#### Scenario: HTML evidence QA
- **WHEN** 生成或验证静态 HTML evidence report
- **THEN** 检查 MUST 验证关键 section、空状态、candidate/pending caveat、HTML escaping 和无远程 CDN/脚本依赖
- **AND** 检查 MUST 不启动常驻 web server

#### Scenario: 表格和图数据 QA
- **WHEN** 生成或验证 paper table、CSV summary、figure data 或 final checklist
- **THEN** 检查 MUST 验证必需列、非空关键数据、method/reference 字段、comparability 字段和 caveat 字段
- **AND** 检查 MUST 拒绝把 candidate-only、pending、mock/smoke 或 not-comparable 行写成 reviewed claim

### Requirement: 实验预算 manifest
长时间训练、fresh evaluation、cache rebuild、checkpoint 写入或多 seed sweep 前，项目 MUST 支持记录实验预算 manifest 或等价 dry-run summary。预算信息 MUST 足以让维护者判断 GPU/时间/输出/数据/清理风险。

#### Scenario: 长跑前记录预算
- **WHEN** 用户准备启动长时间训练或多 seed sweep
- **THEN** budget manifest MUST 记录 workflow/change id、config/manifest path、dataset family、是否读取真实 dataset、GPU/CPU 需求、预计时长、输出 root、checkpoint/cache 计划和停止条件
- **AND** manifest MUST 声明生成产物默认不提交

#### Scenario: 预算缺字段时保守失败
- **WHEN** 长跑命令缺少输出 root、真实数据读取声明、checkpoint 写入声明或停止条件
- **THEN** dry-run 或 preflight SHOULD 报告缺失字段
- **AND** 实现 MAY 要求用户补齐后再启动真实运行

### Requirement: Run recipe 和环境 fallback
项目 MUST 记录可复现的 smoke/dev run recipe，并区分 smoke/dev 环境与 GPU/full training 环境。若 console script 在当前环境不可用，文档或验证 helper MUST 提供包内 `python -m` fallback 以定位安装问题。

#### Scenario: console script 不可用时有 fallback
- **WHEN** `conda run -n kd_mm_beam kd-sensing-<entry>` 因 PATH 或 editable install 问题不可用
- **THEN** 文档或诊断输出 SHOULD 提示等价 `conda run -n kd_mm_beam python -m kd_sensing.cli.<owner>` 检查路径
- **AND** 该 fallback MUST 不成为绕过 package CLI 的长期新入口

#### Scenario: 环境 recipe 不包含本地秘密
- **WHEN** 项目记录 smoke/dev 或 GPU/full training 环境说明
- **THEN** recipe MUST 不包含本地数据绝对路径、密码、token、平台内部启动配置或 checkpoint 文件
- **AND** 如需真实数据路径，MUST 通过用户显式参数或 ignored local config 提供

### Requirement: Research run preview 验证
项目 MUST 为 research run preview loop 提供 focused validation，覆盖 happy path、preview QA、budget manifest 和 run recipe。验证 MUST 使用 `kd_mm_beam` 环境运行项目 Python 命令。

#### Scenario: focused validation 无真实数据依赖
- **WHEN** 开发者运行 research run preview focused tests
- **THEN** 测试 MUST 使用临时目录和 synthetic fixture
- **AND** 测试 MUST 不读取真实 `dataset/`、不加载真实 checkpoint、不写入源码内产物

#### Scenario: 文档入口保持短链路
- **WHEN** README 或 agent navigation 记录 research run preview happy path
- **THEN** 文档 MUST 保留短命令和链接
- **AND** 详细字段、预算 schema 和 QA 规则 MUST 放在 `docs/` 或 OpenSpec 中
