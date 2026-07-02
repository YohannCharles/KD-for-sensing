## Context

仓库已经有 `project-surface-cleanup`、`project-health-guardrails` 和 `component-registry` 等治理规格，且近期已经做过多轮项目表面瘦身。Ponytail 审计指出的剩余问题不是单点 bug，而是跨越源码、测试、脚本、文档和注册表的低价值维护面：一部分应从源码删除，一部分应改为直接 owner 导入，一部分应由护栏防止回流。

当前工作树还存在与本 change 无关的归档和脚本改动。实现本 change 时必须保持边界清晰，只处理本方案列出的文件和必要测试，不回滚或覆盖用户已有改动。

## Goals / Non-Goals

**Goals:**

- 删除已经有替代路径、且没有当前消费方的跟踪文件和历史表面。
- 收缩内部 package facade，使内部源码和测试直接导入 owner 模块。
- 让 component registry 的 removed guard 回到“只服务当前迁移”的用途。
- 保持 CLI、公开工作流和评估输出语义稳定。
- 用最小的 inventory、测试或健康检查防止同类低价值表面重新进入仓库。

**Non-Goals:**

- 不清理 `dataset/`、`outputs/`、`logs/`、checkpoint、历史权重或实验运行结果。
- 不删除当前仍作为公开 CLI/API 兼容面的 facade，例如已有规格保护的 JEPA benchmark facade。
- 不重构模型、训练流程、配置加载或评估结果格式。
- 不新增依赖，也不引入新的脚本框架。

## Decisions

1. **按证据分波次删除，而不是一次性大扫除。**  
   每个候选先记录调用方、owner 替代路径、公开 API 风险、inventory/OpenSpec 消费关系和最小验证命令。替代方案是直接按审计清单删除全部候选，但这会把本地 artifact、公开 facade 和源码表面混在一起，容易误删。

2. **严格区分“源码删除”和“本地 artifact 清理”。**  
   `legacy_knowledge_decoupling_cleanup_manifest.json` 是已跟踪根目录历史清单，属于源码表面，应删除或移到可再生成的本地产物位置；`src/kd_sensing.egg-info/` 是 ignored 生成物，只能作为本地清理动作，不应出现在源码 diff 中。替代方案是把两者都作为代码变更处理，但这会污染 review 边界。

3. **package facade 采用直接 owner 导入，不新增兼容聚合层。**  
   对 `src/kd_sensing/data/mmw/__init__.py`、`engine/__init__.py`、`utils/__init__.py`、`preprocessing/__init__.py`、`evaluation/__init__.py`、`losses/__init__.py`、`models/physics/__init__.py`、`baselines/rmbp_mm/__init__.py` 等低价值重导出，内部调用点改为具体模块导入，`__init__.py` 保留轻量 marker 或明确公共入口。替代方案是保留 deprecated wrapper，但这会继续扩大导入面，且没有当前收益。

4. **removed registry guard 只保留迁移价值。**  
   保留对当前用户仍可能遇到、且有明确替代方向的旧名称诊断；删除只服务历史 fixture 或旧实现变体的 `register_removed` 项，相关测试改为验证 canonical 注册和普通 unknown-name 诊断。替代方案是继续维护完整 tombstone 表，但它会变成被测试保活的历史索引。

5. **未登记脚本默认不提交，复用价值才合并到现有入口。**  
   `scripts/run_priority_v3_budget.sh` 作为未跟踪一次性脚本，不应直接进入 current surface。若实现前确认其中有唯一、可复用的调度逻辑，应折叠到 `scripts/run_next_v3_experiments.sh` 或 Python 工作流入口，并同步 inventory；否则删除本地文件即可。

6. **U-Mask Beam JEPA 指标去重保持输出不变。**  
   缺失矩阵评估中的 top-k、ADBA、MAE 应从同一次 `beam_classification_circular_summary` 派生，删除重复 top-k helper 路径或停止在该 workflow 中使用它。替代方案是保留两个实现并靠测试发现漂移，但这增加了无收益分叉。

## Risks / Trade-offs

- **外部代码依赖包级重导出** → 只删除审计证据显示内部无当前消费、且已有 owner 模块替代的 facade；保留明确公开入口，并在变更说明中列出导入迁移方式。
- **registry 错误消息测试过度绑定历史名称** → 测试改为验证当前迁移 guard 和 canonical 行为，低价值旧名允许回落到普通未知组件错误。
- **未跟踪脚本含有未记录的实验知识** → 删除前快速阅读脚本；如果有唯一参数组合或预算策略，合并到现有登记入口或压缩进 inventory 说明。
- **清理本地 artifact 被误解为源码变更** → `src/kd_sensing.egg-info/` 只作为本地删除动作记录，不纳入 git diff；验证时不依赖它存在或不存在。
- **与当前脏工作树冲突** → 实现前后查看 `git status --short`，只改本 change 需要的文件，不触碰已归档 change 的删除/新增状态。

## Migration Plan

1. 建立基线：记录 `git status --short`、候选文件跟踪状态、关键 facade 调用方、registry removed guard 分类。
2. 删除源码表面：移除根目录历史清理清单和无当前价值的根目录笔记，更新 inventory。
3. 收缩 package facade：改内部导入为 owner 模块，保留轻量 marker 或必要公共入口。
4. 精简 registry removed guard：删除低价值 tombstone 和镜像 fixture 断言，保留有迁移说明的 guard。
5. 处理脚本表面：不提交未登记脚本；若合并唯一逻辑，则同步当前脚本 inventory 和对应测试。
6. 去重评估指标：让 U-Mask Beam JEPA 缺失矩阵从单一 circular summary 派生 top-k/DBA/MAE。
7. 增补护栏和 focused 测试，最后运行 OpenSpec 和相关 pytest。

回滚策略按波次执行：若某一波引入导入或行为风险，只回退该波文件；删除文件均有 owner 替代或可从 git 历史恢复。

## Open Questions

无阻塞问题。`scripts/run_priority_v3_budget.sh` 是否包含唯一逻辑将在实现前阅读后决定“删除”或“折叠进现有入口”，默认不作为新 current surface 保留。
