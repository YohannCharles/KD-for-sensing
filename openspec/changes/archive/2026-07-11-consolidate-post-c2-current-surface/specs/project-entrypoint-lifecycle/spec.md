## ADDED Requirements

### Requirement: Temporal 和历史 launcher 不得派生平行 script suite
H5/P1 temporal matrix 已覆盖的 check/launch/eval/summary 行为 MUST 通过现有参数化脚本使用。项目 MUST 不保留通过 `sys.path` 注入、脚本私有函数导入或模块全局变量改写派生的 S1-S4 parallel wrappers；历史 overnight launcher 在结果冻结后 MUST 退出 current script surface。

#### Scenario: S1-S4 wrapper 删除
- **WHEN** temporal router S1-S4 tasks 被 defer
- **THEN** 三个 S1-S4 wrapper 和专属 tests MUST 删除
- **AND** H5/P1 launcher 的用户改动 MUST 保留

#### Scenario: 历史 launcher 退出但 summary 保留
- **WHEN** overnight training matrix 只剩历史结果复盘价值
- **THEN** launcher MUST 删除
- **AND** 仍被 final C2 summary 消费的 read-only summary helper MAY 保留

## MODIFIED Requirements

### Requirement: Viewer manifest 聚合模块已退役
Viewer manifest 相关模块与 wrapper MUST 继续退役。该 guard MUST 不再把 JEPA visual analysis 或 GPS shortcut benchmark 作为 current migration owner；current diagnostics/evaluation 只指向 U-Mask matrix、MMW/CSI、Scene31-34 final analysis 和其它明确 retained owner。

#### Scenario: Viewer helper 不存在
- **WHEN** diagnostics/CLI surface 被检查
- **THEN** viewer manifest helpers/prediction exporter MUST 不存在
- **AND** 不得迁移到已退役 JEPA diagnostics

### Requirement: 退役旧模态诊断脚本入口
旧 modality subset/perturbation scripts MUST 不作为长期入口。通用 subset/mask/difficulty 行为 MUST 由 shared evaluation、U-Mask matrix、missing-stress、MMW/CSI 或内部 helper 承载；JEPA visual/shortcut MUST 不在 allowlist。

#### Scenario: Script allowlist 使用 current owners
- **WHEN** architecture test 枚举 scripts/tools
- **THEN** 旧 modality scripts 和 JEPA visual/shortcut wrappers MUST 不存在
- **AND** retained dataset preparation、MMW/CSI、U-Mask 和 Scene31-34 entries MAY 保留

#### Scenario: 通用 subset 能力保留
- **WHEN** current evaluation 配置启用 modality subset/mask
- **THEN** shared evaluation MUST 继续工作
- **AND** 不依赖 retired scripts

### Requirement: 退役入口回流必须被架构边界测试拒绝
架构边界测试 MUST 拒绝 retired CLI/module/script/config/wrapper 回流，并 MUST 将迁移方向指向 final C2/U-Mask、retained package CLI、MMW/CSI、Scene31-34 或普通 unknown-name behavior；不得指向 JEPA visual/shortcut 等本轮删除 owner。

#### Scenario: 旧模块与脚本不回流
- **WHEN** structure guard 运行
- **THEN** retired module/script/config tokens MUST 不作为 current surface存在
- **AND** historical docs MAY 保留明确 retired wording

### Requirement: 当前推荐 workflow 排除 Top8 residual coarse 路线
README、quickstart、experiment matrix 和 inventory MUST 将 current workflow 聚焦 final C2/U-Mask、retained train/evaluate/preprocess、MMW/CSI、AMR/AMBER controls、Scene31-34 evidence 和必要 supporting owners。Top8/residual/BGAM/viewer、Image+GPS query、Vision-Position、JEPA visual/shortcut 和 geometry MUST 不作为 current 推荐面。

#### Scenario: Quickstart 使用 retained workflow
- **WHEN** current docs 被检查
- **THEN** 推荐命令 MUST 来自十个 package CLI 或 protected local/manual owner
- **AND** retired command/config MUST 只在 historical context出现

#### Scenario: Guard 不要求 retired imports
- **WHEN** quick health check 运行
- **THEN** 它 MUST 不导入 retired route
- **AND** MAY 断言 retired route不存在

### Requirement: Scene31/Scene31-34 报告脚本必须分类为本地研究报告表面
Scene31-34 final analysis 的论文表格、per-scene summary、profile 和 final conclusion MAY 保留为 research diagnostic 或 local/manual reporting surface，并 MUST 在 inventory 或 current 文档中登记 lifecycle、职责和输出边界。旧 Scene31 baseline-pack/next-round/shared summary 已退役，MUST 不再作为 current report surface。

#### Scenario: Scene31-34 报告 owner 有输出边界
- **WHEN** 项目保留 Scene31-34 final analysis owner
- **THEN** inventory MUST 说明其读取本地 summary、fresh-eval 或 paper table 输入
- **AND** 输出边界 MUST 限定在 ignored `outputs/`、`logs/` 或显式用户路径

#### Scenario: 旧 Scene31 report 不升级为 package CLI
- **WHEN** README、AGENTS、OpenSpec 或 docs 描述当前推荐入口
- **THEN** 它们 MUST 不推荐旧 Scene31 baseline-pack/next-round/shared summary
- **AND** protected Scene31-34 local/manual owner MUST 不被描述为 package CLI

### Requirement: Wrapper 删除需要 focused guardrail
删除 wrapper 后，项目 MUST 通过轻量 architecture boundary 或 focused tests 防止同职责 wrapper 回流。保留的 scripts MUST 有明确 owner、输入输出边界和删除条件，且 package CLI MUST 不依赖 local/manual script。

#### Scenario: 结构检查拒绝 wrapper 回流
- **WHEN** 开发者运行 architecture boundary 或 compile check
- **THEN** 检查 MUST 拒绝 `sys.path` script-to-script import、模块全局 monkeypatch 和只转发默认参数的 wrapper
- **AND** 检查 MUST 不要求 scripts lifecycle doctor 存在

### Requirement: Post-C2 public CLI 必须收敛到主线、MMW 和治理入口
项目在 post-C2 清理后 MUST 只声明十个 public console scripts：train、evaluate、preprocess、runs、runtime cleanup、runtime organize、paper export、U-Mask eval matrix、MMW GPS v2 和 MMW physics inspect。Research dashboard/preview、project surface doctor、architecture summary、training throughput、dataset/source audit 和历史复现 CLI MUST 不再作为 public console script。

#### Scenario: 删除 CLI 同步所有 current references
- **WHEN** implementation 从 pyproject 删除 dashboard、preview 或 surface doctor
- **THEN** README、docs、current specs、CLI help smoke 和 inventory MUST 同步删除 current command reference
- **AND** 删除后项目 MUST 不提供同名 console script、module alias 或 thin wrapper

#### Scenario: 保留 CLI 有生命周期锚点
- **WHEN** post-C2 清理完成
- **THEN** 十个保留命令 MUST 在 pyproject 与 inventory 中有 owner、输出边界和 focused validation
- **AND** 它们 MUST 不依赖已删除 script、dashboard 或 historical config

## REMOVED Requirements

### Requirement: Scene31 local/manual 入口必须统一生命周期
**Reason**: 该 requirement 整体描述已退役的 Scene31 next-round、BC、funnel 和 magic overnight workflow；保留 workflow 已迁移到独立 Scene31-34 owner。
**Migration**: 使用 `scenes31-34-main-missing-modality-workflow` 与更新后的 Scene31-34 reporting lifecycle requirement。

#### Scenario: 旧 Scene31 local/manual surface 退出
- **WHEN** scripts、inventory 和 current docs 被枚举
- **THEN** Scene31 next-round/BC/funnel/magic runner、generator 和 summary MUST 不再是 current surface
- **AND** Scene31-34 protected runner/generator/final analysis MUST 保持

### Requirement: Scripts lifecycle doctor
**Reason**: 1,447 行 surface doctor 未能发现当前 S1-S4 script-to-script wrappers，且与 pyproject、inventory 和 architecture tests 重复。
**Migration**: 使用现有 architecture boundary、CLI/config verify 和 compile checks 验证结构事实。

#### Scenario: 未分类 script 由结构检查发现
- **WHEN** tracked script 缺少 inventory 分类或引用不存在 config
- **THEN** focused structure check MUST 失败
- **AND** 项目不再提供 doctor report 产品面

### Requirement: HTML dashboard 入口保持只读诊断
**Reason**: Research dashboard 与 HTML renderer 整体退役，不再需要只读 dashboard CLI 生命周期。
**Migration**: 使用 run index、人工 claim registry 和 paper export 获取保留证据。

#### Scenario: Dashboard 命令不可用
- **WHEN** 用户请求旧 research dashboard 命令
- **THEN** console script MUST 不存在
- **AND** 文档 MUST 指向保留的证据 owner

### Requirement: Scene31 重复 wrapper 必须收敛到 canonical command
**Reason**: 旧 canonical `scene31_summary` owner 本身已无 current consumer 并随 retired Scene31 capability 删除。
**Migration**: 历史结果通过 docs、claim notes、OpenSpec archive 和 git 查询；current Scene31-34 final analysis owner保持不变。

#### Scenario: 旧 Scene31 summary 命令不可用
- **WHEN** 用户请求 `python -m kd_sensing.diagnostics.scene31_summary`
- **THEN** module-only CLI MUST 不再属于 current surface
- **AND** current docs MUST 不推荐其 profile 或 wrapper
