## Context

当前工作树已经完成多轮治理和架构收口：OpenSpec active change 列表为空，多个 change 已进入 `openspec/changes/archive/`，Scene31 大量生成型 YAML 被删除并由 manifest/generator 维护，JEPA diagnostics、training/data/runtime hotspot 也被拆到更窄的 owner 模块。快速验证结果显示核心状态健康：`openspec validate --all --strict`、架构边界测试、Scene31 focused tests、核心 focused tests 和全量 pytest 均可通过。

剩余问题不是功能行为错误，而是归档后治理表面的小漂移：

- `docs/maintainer_context_index.yaml` 仍记录一个已不存在的 change-specific validation 命令，直接运行会失败。
- `docs/project_surface_inventory.md` 的数量基线没有明确区分 tracked-only、当前工作树 on-disk 和历史 CodeGraph/AST baseline，容易被误读为硬预算。
- 新拆分测试中的 helper 导入仍通过普通测试文件头部插入 `tests/` 路径实现，虽然功能可用，但与 shared pytest bootstrap 收口方向不一致。
- OpenSpec archive 与删除 active change 目录需要提交完整性审计，避免只提交一半状态。
- MMW helper 里 pandas 逐列插入触发 fragmentation warning，属于可选性能/噪声清理。

本 change 只处理文档、OpenSpec、架构边界测试和无语义性能清理，不改变训练、评估、预处理、模型 forward、数据 split、checkpoint schema 或 runtime output 语义。

## Goals / Non-Goals

**Goals:**

- 让维护索引中的 validation 命令全部对应当前可运行的 OpenSpec 或 pytest 检查。
- 让 inventory 的统计基线明确声明来源、范围、排除项和用途，避免用数量漂移替代架构判断。
- 让普通测试通过 shared bootstrap 或 package-style helper import 复用测试 helper，移除文件级 `tests/` path 注入。
- 让架构边界测试能发现归档后只提交删除或只提交 archive 的不完整状态。
- 可选修复 MMW helper 的 pandas fragmentation warning，并用 MMW focused tests 验证行为不变。
- 保持所有 Python 验证命令使用 `conda run -n kd_mm_beam ...`。

**Non-Goals:**

- 不新增或删除 package CLI、训练入口、评估入口、预处理入口或诊断入口。
- 不恢复任何已退役 KD/HiST/BGAM/viewer/legacy config 路径。
- 不重新组织 Scene31 workflow 的功能语义，不重新生成或提交 generated YAML。
- 不扩大 `docs/maintainer_context_index.yaml` 为完整源码目录镜像或完整 allowlist 数据库。
- 不把 pandas warning 清理升级为 dataset loader 语义重构。

## Decisions

### 1. 用当前可运行命令替换 change-specific validation

维护索引的 focused validation 不再引用已归档或不存在的 change name。归档后默认使用：

- `openspec validate --all --strict`
- `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py tests/test_scene31_next_round.py -q`
- 按触碰范围追加 config/CLI、training/data/evaluation、JEPA diagnostics 或 MMW focused tests

替代方案是保留历史 change-specific 命令并在旁边注明“历史记录”。该方案会让维护者复制命令后直接失败，不适合作为机器可读 focused validation。

### 2. inventory 统计只作为趋势基线，必须标明口径

inventory 首段应明确使用当前工作树 on-disk 扫描还是 tracked-only 扫描，并说明历史 CodeGraph/AST 基线是过去审计点，不是当前硬预算。实现时优先写清：

- 统计时间或 change 收口点。
- 扫描范围：`src/kd_sensing`、`tests`、`scripts`、`configs`。
- 排除项：`dataset/`、`outputs/`、`logs/`、cache、checkpoint、本地运行产物。
- 用途：趋势定位和右尺寸化上下文，不是自动拆分 KPI。

替代方案是把数字删除，只保留“很多文件”。这会降低 inventory 的审计价值，也不利于后续判断拆分是否来自合理 helper 增长。

### 3. 普通测试 helper 采用 package-style import

拆分后的普通测试 helper 保留在 `tests/` 下，但测试文件通过 `from tests.<helper> import ...` 或等价 shared bootstrap 可解析路径导入。普通测试文件不得继续复制：

```python
TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))
```

架构边界测试保留对子进程 import probe 的例外，因为这类 probe 需要显式构造干净解释器路径。替代方案是把 helper 移到 `src/kd_sensing/testing/`，但这会把测试工具暴露到 runtime package，收益小且容易扩大公开 surface。

### 4. 归档完整性作为 git 状态审计，而非 runtime 行为

归档后完整性检查只读取 `git status --short`、tracked paths 和 `openspec list --json`。当同一 change 同时出现：

- `D openspec/changes/<change>/...`
- `?? openspec/changes/archive/<date>-<change>/...`

架构边界或实施任务必须要求二者成对纳入提交，或在最终说明中记录暂缓原因。该检查不得把 archive 目录当作 active requirement，也不得自动删除或移动用户文件。

替代方案是忽略未跟踪 archive，由人工提交时发现。这个方案已经暴露风险：大量 archive 目录容易在大 diff 中漏加。

### 5. MMW pandas warning 作为可选局部优化

`mmw_columns.py` 若通过循环给 DataFrame 多次插入列，应改为先构造新列 dict/DataFrame，再一次性 concat。验证重点是 shape、字段名、metadata 和现有 MMW sample contract 不变。该改动不进入 OpenSpec 行为要求，只作为任务中的可选实现项。

替代方案是用 warning filter 静默 warning。该方案隐藏真实性能噪声，不如小范围修复列构造。

## Risks / Trade-offs

- [Risk] 架构边界测试过度检查 git status，误伤开发中的普通 archive 草稿。→ Mitigation: 检查只要求成对审计或记录 deferral，不自动失败所有未跟踪 archive；实现时可限制在已删除 active change 且存在同名 dated archive 的场景。
- [Risk] 把测试 helper 改成 `tests.<helper>` import 后，在某些 pytest 启动方式下 `tests` 包不可解析。→ Mitigation: 先确认 `tests/conftest.py` 已将仓库根加入 `sys.path`，并运行拆分后 helper focused tests。
- [Risk] inventory 数字在用户继续新增未跟踪文件时再次变化。→ Mitigation: 文档写“当前工作树 on-disk 扫描”和扫描命令口径，避免承诺永久精确。
- [Risk] MMW warning 修复意外改变 DataFrame index 或列顺序。→ Mitigation: 使用原 frame index 构造新列 DataFrame，并运行 `tests/test_mmw_town10_preparation.py`。
- [Risk] 新 validation 列表过长导致维护者忽略。→ Mitigation: 维护索引保留 focused 分层命令，完整命令矩阵继续放在 inventory/agent navigation，不把索引扩成验证手册。

## Migration Plan

1. 更新 `docs/maintainer_context_index.yaml`，删除不存在 change 的 validation 命令，改为当前可运行的 OpenSpec/spec/pytest 分层命令。
2. 更新 `docs/project_surface_inventory.md` 和必要的导航文档，声明统计口径、归档完整性和测试 helper 导入边界。
3. 扩展 `tests/test_architecture_boundaries.py`：
   - 检查维护索引中的 `openspec validate <change>` 不引用 inactive/missing change。
   - 检查普通 `tests/test_*.py` 不维护文件级 `tests/` path insertion。
   - 检查 inventory 统计段包含口径/范围/排除项关键标记。
   - 检查 OpenSpec archive 删除/新增成对状态或可记录 deferral。
4. 修改拆分后的测试文件，将 helper import 改为 package-style import。
5. 可选修改 `mmw_columns.py` 的 DataFrame 列构造方式。
6. 运行分层验证；若 MMW warning 修复导致任何 focused test 失败，回滚该局部优化，不影响治理文档和测试收口。

Rollback 策略：本 change 的核心是文档和测试 guardrail，可逐文件回滚；MMW warning 修复独立于治理规则，失败时可单独撤回。

## Open Questions

- 架构边界测试是否应强制所有 OpenSpec archive 成对提交，还是只在 active change 删除与同名 archive 新增同时出现时检查？默认选择后者，以降低开发期误伤。
- inventory 统计是否采用当前工作树 on-disk 口径还是 tracked-only 口径？默认采用 on-disk 口径并说明 tracked-only 可能不同，因为本轮大变更已有大量新增 helper 尚未进入 git index。
- `docs/maintainer_context_index.yaml` 是否继续保存 focused validation 列表，还是只指向 `docs/project_surface_inventory.md`？默认保留最小 focused 列表，避免维护索引失去机器可读价值。
