## ADDED Requirements

### Requirement: 静态 HTML evidence dashboard
系统 MUST 提供静态 HTML evidence dashboard，用于把 research dashboard summary 渲染为可离线打开的人类可读页面。HTML dashboard MUST 与文本摘要和 JSON 输出使用同一 summary 数据，不得重新扫描运行目录或重新计算 claim gate。

#### Scenario: 渲染完整 dashboard 页面
- **WHEN** dashboard summary 包含 active changes、run states、resources、claim candidates、paper readiness、warnings 和 next actions
- **THEN** HTML 输出 MUST 展示这些 section 或等价摘要
- **AND** 页面 MUST 标记 candidate-only、draft、pending、unverified、not_comparable 或 mock/smoke 状态
- **AND** 页面 MUST 不把候选 claim 写成已审阅论文结论

#### Scenario: 空输入仍生成可读页面
- **WHEN** summary 中没有 active change、running run、claim candidate 或 paper gate rows
- **THEN** HTML 输出 MUST 生成完整页面
- **AND** 页面 MUST 显示空状态、generated_at 或等价 metadata
- **AND** 命令 MUST 不因为空输入失败

### Requirement: HTML dashboard 输出边界
HTML dashboard MUST 只写入 ignored output root 或用户显式路径。生成 HTML MUST 不移动、删除、压缩、复制或重写 `dataset/`、`outputs/`、`logs/`、checkpoint、cache、ledger 原始记录或 current docs。

#### Scenario: 写入显式 HTML 路径
- **WHEN** 用户通过 CLI 指定 HTML 输出路径
- **THEN** 系统 MUST 创建必要父目录并写出 `.html` 文件
- **AND** 输出提示 MUST 包含 HTML 路径
- **AND** HTML 文件 MUST 只包含 dashboard summary 的可读摘要和本地 artifact 路径引用

#### Scenario: 不修改正式 claim 文档
- **WHEN** dashboard 发现 upgradable candidate 或缺失 evidence
- **THEN** HTML 输出 MUST 只展示 next action 和 caveat
- **AND** 系统 MUST NOT 修改 `docs/result_claims_registry.md`、README、OpenSpec specs 或实验文档

### Requirement: 离线安全渲染
HTML dashboard MUST 离线可打开，并 MUST 对来自本地运行产物、路径、日志、claim id、warning、hint 和 OpenSpec change 名称的动态文本进行 HTML escaping。HTML dashboard MUST 不依赖外部 CDN、远程 JavaScript、远程字体或网络请求。

#### Scenario: 特殊字符安全展示
- **WHEN** summary 字段包含 `<script>`、HTML 标签、引号、反斜杠、中文、路径空格或 shell 字符
- **THEN** HTML 输出 MUST 将这些内容作为文本展示
- **AND** 页面 MUST 不产生可执行脚本注入

#### Scenario: 离线页面无远程依赖
- **WHEN** 用户在无网络环境打开生成的 HTML
- **THEN** 页面 MUST 保持主要内容可读
- **AND** HTML MUST 不引用 `http://`、`https://`、远程 CDN 或外部 JavaScript

### Requirement: HTML dashboard 验证
HTML dashboard MUST 具备 focused tests，覆盖 renderer、CLI 输出和安全边界。测试 MUST 使用临时目录和 synthetic summary，不得读取真实 `dataset/`、真实 checkpoint 或用户本地训练产物。

#### Scenario: focused tests 覆盖 renderer 和 CLI
- **WHEN** 开发者运行 dashboard focused tests
- **THEN** 测试 MUST 验证 HTML 文件生成、关键 section 存在、candidate-only caveat 存在、动态文本被 escaping、空 summary 可渲染
- **AND** CLI help smoke MUST 覆盖新增 HTML 参数
