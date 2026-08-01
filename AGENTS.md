# AGENTS.md

本文件约束 AI Agent 和协作者在本仓库中的操作方式。

项目结构、架构契约和需求定义以 OpenSpec、README 及 current specs 为准。本文件只记录工作规则，不维护完整目录清单。模型、推理强度和子 Agent 并发数量由 `.codex/config.toml` 管理。

## 基本原则

- 使用简体中文撰写说明、计划、OpenSpec 产物、任务总结和提交相关描述。
- 代码、变量、类、函数和外部接口遵循项目现有英文命名风格。
- 先阅读已有实现、README、相关 OpenSpec 和测试，再修改代码。
- 优先沿用现有模块边界、配置风格、命名规则和错误处理方式。
- 修改前先理解真实调用链和数据流，不要只根据文件名或局部搜索结果推断行为。
- 不要为了代码整洁而顺手重构任务范围外的模块。
- 不要删除、覆盖或回退用户已有但与当前任务无关的修改。
- 不要在未实际验证的情况下声称问题已经解决、测试已经通过或性能已经提升。
- 除非用户明确要求，不要自行创建 commit、tag、分支或执行远程推送。
- 禁止擅自执行 `git reset --hard`、`git clean -fd`、强制切换分支、强制推送等破坏性 Git 操作。
- 不要把本地数据、训练输出、日志、缓存、checkpoint 或临时验证产物纳入源码变更。
- 不要新增旧入口、兼容聚合层或绕过当前 `src/kd_sensing` 包结构的运行方式。
- 遇到需求歧义、实现冲突或信息不足时，应明确说明，不要自行虚构项目约定。

## 上下文加载

非平凡改动前先阅读：

- `docs/agent_navigation.md`
- `docs/maintainer_context_index.yaml`
- 与任务相关的 current specs
- 对应的 active OpenSpec change
- 相关实现和测试

通过 `docs/maintainer_context_index.yaml` 确认任务路由、机器可读治理表和最小验证命令。

若任务匹配 scoped context，按照 `docs/agent_context/README.md` 只加载相关上下文，避免无目的扫描整个仓库。

高频模型、配置和诊断流程可以使用 `.codex/skills/kd-*/SKILL.md` 项目技能。技能只描述流程，不替代 OpenSpec、README、current specs 或本文件中的安全规则。

## 多 Agent 协作

主 Agent 是当前任务的协调者和最终责任人，负责理解需求、任务拆分、架构决策、结果审查、最终整合和验证。

子 Agent 仅负责边界清晰的探索、实现、测试、日志分析或独立审查任务。

### 使用原则

- 平凡且范围明确的修改由主 Agent 直接完成，不要为了使用多 Agent 强行拆分。
- 非平凡任务优先先做并行只读探索，再制定实施方案。
- 可以并行安排子 Agent：
  - 定位相关入口、实现、调用链和数据流；
  - 查找类似实现和可复用模式；
  - 分析相关测试、边界条件和最小验证命令；
  - 检查架构、兼容性、数据隔离和潜在回归；
  - 分析实验配置、日志和评估结果。
- 子 Agent 返回关键结论后，由主 Agent 核实源码、配置或测试，再决定最终方案。
- 不要让多个子 Agent 重复扫描完全相同的范围，除非需要独立交叉验证。
- 子 Agent 不应自行继续派生大量下级 Agent，除非主 Agent 明确授权。

### 修改规则

- 可以把互不重叠的实现任务交给不同子 Agent。
- 禁止多个 Agent 同时修改同一个文件。
- 涉及同一核心文件、公共接口或跨模块协调的改动，由主 Agent 统一修改或串行处理。
- 子 Agent 不得擅自扩大任务范围、改变既定架构方向或修改无关模块。
- 子 Agent 发现任务范围外的问题时，应报告主 Agent，不要顺手大范围修复。
- 子 Agent 不得擅自修改 OpenSpec 所确定的方向；发现实现与 OpenSpec 冲突时，应交由主 Agent 决策。
- 子 Agent 不得执行破坏性 Git 操作、污染系统配置或启动未获授权的长时间训练。
- 实现完成后，可使用独立子 Agent 检查正确性、测试覆盖、兼容性、数据泄漏和潜在回归。

### 子 Agent 返回内容

子 Agent 应简要返回：

- 检查或修改的文件；
- 关键类、函数、配置和调用路径；
- 主要发现或实际改动；
- 执行的命令和验证结果；
- 未执行的验证及原因；
- 风险、不确定性和需要主 Agent 决策的问题。

主 Agent 不得盲目采纳子 Agent 输出。影响最终实现的重要判断必须结合源码、配置、测试或实验记录核实。

## OpenSpec

当前架构和需求权威位于 `openspec/specs/`：

- 模型与保留路线：`u0-mainline`
- 数据隔离：`clean-data-integrity`
- 仓库与产物边界：`repo-boundaries`

有 active change 时，先阅读对应内容：

```text
openspec/changes/<change>/proposal.md
openspec/changes/<change>/design.md
openspec/changes/<change>/tasks.md
openspec/changes/<change>/specs/
```

以下变化通常应先创建或更新 OpenSpec change：

* 非平凡功能；
* 架构调整；
* 模型结构变化；
* 训练流程变化；
* 数据契约变化；
* CLI、配置系统或公共接口变化；
* 兼容性变化；
* 正式实验方案或论文 claim 变化。

实现过程中发现范围、需求或设计决策变化时，应先更新对应 OpenSpec artifact，再继续修改代码。

没有 active change 且只是窄修复、文档小改、测试修复或不改变契约的内部实现修正时，可以直接修改，但仍需遵守现有 specs。

## 命令环境

所有项目相关 Python 命令、测试、训练、验证和依赖安装都必须使用 `kd_mm_beam` 环境：

```bash
conda run -n kd_mm_beam <command>
```

常用入口：

```bash
conda run -n kd_mm_beam kd-sensing-train --help
conda run -n kd_mm_beam kd-sensing-evaluate --help
conda run -n kd_mm_beam kd-sensing-preprocess --help
```

不要直接使用系统环境执行：

```bash
python ...
pytest ...
pip install ...
```

应使用：

```bash
conda run -n kd_mm_beam python ...
conda run -n kd_mm_beam pytest ...
conda run -n kd_mm_beam python -m pip install ...
```

除非用户明确要求，不要擅自升级或批量修改依赖版本。安装新依赖前，应先检查现有依赖文件、锁文件和项目约定。

## 系统配置与启动项安全

* 禁止修改、重写或把训练命令写入容器启动、认证或凭证配置文件，例如：

  * `/root/.container_env`
  * `/etc/profile`
  * `/etc/environment`
  * SSH 配置
  * 系统账号密码配置
  * 平台凭证文件
* 只有用户明确点名要求修改具体文件时，才可以考虑相关修改。
* 修改前必须说明风险、备份原文件并展示拟修改内容。
* `/root/.container_env` 只能保留容器平台写入的运行状态、用户名、密码、IP、网关等字段。
* 严禁把 `cd ...`、`CUDA_VISIBLE_DEVICES=...`、`kd-sensing-train ...`、`nohup ...`、`tmux ...` 等命令写入 `USERNAME`、`PASSWD` 或其他凭证字段。
* 长时间训练优先使用当前 shell、`conda run`、`tmux`、`nohup`、项目脚本或平台任务系统。
* 不要通过污染系统凭证文件实现所谓“开机自启”。
* 如确实需要配置自启动，必须先获得用户明确确认，并使用专门的启动机制或进程管理工具。

## 训练与实验安全

启动长时间训练或批量实验前，必须确认：

* 配置文件；
* 数据集和场景划分；
* 历史窗口和预测窗口；
* 随机种子；
* GPU 分配；
* 输出目录和日志目录；
* 是否覆盖已有结果；
* 恢复训练还是全新训练；
* 评估指标和缺失模式；
* 预计运行数量。

并行实验时：

* 每个任务使用独立输出目录和日志文件；
* 明确每个任务对应的 GPU；
* 记录配置、seed、命令和运行状态；
* 不要因为单次失败而无限自动重跑；
* 不要静默跳过失败样本、失败测试或缺失指标；
* 不要覆盖已有 checkpoint 或正式实验结果。

涉及实验对比时，必须确保：

* 使用相同数据划分；
* 使用相同训练预算；
* 使用相同预处理和评估协议；
* 使用相同指标实现；
* 使用一致的缺失模式定义；
* 不使用未来信息；
* 不使用测试集调参；
* 不引入未声明的额外标签；
* 不只选择有利 seed 汇报。

## 验证

窄改动优先运行与修改直接相关的测试。

涉及架构、导入边界、CLI、配置或公共工作流时，至少考虑：

```bash
make verify-quick
make verify-cli-config
make verify-compile
```

底层命令保持为：

```bash
openspec validate --all --strict
conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q
conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_config_load_characterization.py -q
conda run -n kd_mm_beam python scripts/verify_compile.py
```

OpenSpec 相关变更还应运行：

```bash
openspec validate <change> --strict
openspec status --change <change>
```

涉及跨模块修改、公共接口变化、架构调整或准备形成正式结果时，应考虑最终回归：

```bash
make verify-full
conda run -n kd_mm_beam pytest -q
```

最终验证由主 Agent 负责。子 Agent 可以运行局部测试，但主 Agent 必须基于最终工作树核对验证结果。

测试失败时：

* 不要直接删除失败测试；
* 不要通过放宽断言掩盖真实问题；
* 不要为了通过测试而改变无关行为；
* 应区分实现错误、测试错误、环境问题和仓库已有失败；
* 应明确区分本次修改导致的失败和历史失败。

未实际运行的测试不得描述为通过。受时间、资源或环境限制而未执行的验证，必须在最终报告中明确说明原因和剩余风险。

## 文档边界

* README 保留安装、快速上手、简短目录结构和主要工作流说明。
* OpenSpec 记录需求、架构约束、设计决策和变更历史。
* 本文件记录 Agent 操作规则，不重复维护完整项目结构。
* `docs/agent_navigation.md` 负责 Agent 导航。
* `docs/maintainer_context_index.yaml` 负责机器可读任务路由和治理信息。
* scoped context 放在 `docs/agent_context/`。
* 正式实验 claim 应记录在对应规范文档或实验报告中，不要只保留在临时聊天或日志中。
* 不要使用 hook 自动重写 README、OpenSpec、本文件或正式 claim 文档。
* 如需防止文档漂移，优先添加检查型脚本或测试，由人决定正式文档如何更新。
* 不要把未经验证的实验结果或性能数字写入正式文档。

## 代码与数据边界

* 优先进行最小、直接、可验证的改动。
* 不进行与当前目标无关的大范围格式化或重命名。
* 不引入不必要的抽象层。
* 不复制已有实现形成第二套并行逻辑。
* 不在测试中复制生产逻辑以制造表面通过。
* 不硬编码本地绝对路径、用户名、GPU 编号或机器特定信息。
* 不在源码中记录密码、Token、私钥或其他凭证。
* 修改公共行为时必须考虑已有调用方和向后兼容性。

必须遵守 `clean-data-integrity` 相关约束，尤其禁止：

* 训练集、验证集和测试集泄漏；
* 使用未来帧预测过去或当前目标；
* 使用推理期不可获得的标签信息构造特征；
* 使用测试集选择超参数；
* 根据测试结果反复修改训练方案；
* 静默改变数据划分或预处理；
* 在不同方法间使用不一致的评估协议。

## 产物边界

以下内容属于本地数据或运行产物，默认不提交：

```text
dataset/
outputs/
outputs/cache/
logs/
cache/
TensorBoard 产物
新生成的 checkpoint
临时分析结果
临时导出文件
```

具体规则：

* `dataset/` 是本地数据输入，默认不提交。
* `outputs/`、`outputs/cache/`、`logs/` 和 legacy 根目录 `cache/` 默认不提交。
* 新可再生成 cache 默认归入 `outputs/cache/`。
* `All_models/` 中已跟踪权重是历史复现实验资料。
* 新生成的 `.pth`、`.pt`、`.ckpt` 不应进入源码变更。
* 提交或汇报前检查是否意外加入大文件、日志、缓存、数据或模型权重。

## 最终报告

任务结束时，主 Agent 应使用简体中文简要报告：

* 实际完成的内容；
* 关键设计决策；
* 修改的主要文件；
* 已执行的验证命令和结果；
* 未执行的验证及原因；
* 已知限制、剩余风险和需要用户决定的问题。

最终报告必须明确区分：

* 已完成；
* 已验证；
* 未验证；
* 仅建议；
* 仍不确定。

不要使用“应该没问题”“大概率通过”等表达替代实际验证结果。
