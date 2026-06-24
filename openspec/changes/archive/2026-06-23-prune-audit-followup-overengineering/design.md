## Context

当前仓库已经通过多轮 right-size/prune change 删除了旧训练脚本、BeamBench 大聚合 owner、builder facade、objective registry/history 包装和一批退役研究线入口。剩余问题更细碎：一些 package `__init__.py` 为方便导入 eager re-export 重模块；少数文件只是单用途包装；诊断和报告模块复制了 JSON/CSV/float helper；registry 仍维护一张长期 removed-name table。

这些问题不影响数值正确性，但会增加导入副作用、迁移表维护成本和后续开发的“顺手加一层”倾向。本 change 以删除和收缩为主，不引入新抽象，不新增依赖，不改变用户运行入口。

## Goals / Non-Goals

**Goals:**

- 减少低价值公开聚合面，内部实现直接导入 owner module。
- 删除或合并单用途包装文件，保留必要的轻量 public path。
- 收敛重复 helper，但只在确有多处复用时创建窄 helper。
- 简化 registry removed-name 维护，只保留仍有当前迁移价值的拒绝说明。
- 增加 focused 架构护栏，防止 barrel/facade/helper 回流和 tracked runtime artifact 污染。

**Non-Goals:**

- 不重构训练主循环、dataset resource loading、diagnostics 大型 workflow 或模型结构。
- 不改变 CLI 名称、配置语义、checkpoint schema、metrics、beam label、数据 split 或本地数据目录策略。
- 不清理 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、`All_models/` 或 ignored 本地产物。
- 不新增全局 `utils` 大杂烩，不把重复 helper 合并成新的跨领域依赖中心。

## Decisions

1. **先迁移内部 import，再删除公开聚合。**
   - 做法：先用静态搜索和 focused tests 找出 `from kd_sensing.<package> import ...` 的内部用法，改成 owner module 路径；确认没有 current public 契约依赖后，再删除或缩短 package barrel。
   - 备选：保留 barrel 并改成 lazy export。拒绝原因：lazy export 仍是额外公开面，容易继续吸收新符号。

2. **registry removed-name table 按迁移价值保留。**
   - 做法：保留仍能帮助当前用户迁移的名称，例如 scene alias 到 `deepsense6g + scene` 的提示；完全退役且已有 config migration guard、retired-tombstone spec 或文档边界覆盖的名称，可移出 registry table，让普通 unknown-name 错误处理。
   - 备选：删除所有 removed guard。拒绝原因：少数历史拼写仍有明确迁移方向，直接 unknown 会降低可诊断性。

3. **重复 helper 先局部合并，确需共享才建窄模块。**
   - 做法：同一 diagnostics family 内重复 `_json_ready`、`_write_csv`、`_float_or_none` 时优先复用已存在 owner helper；跨 family 只有在调用点超过两个且语义一致时，才建类似 `diagnostics/io_helpers.py` 的窄模块。
   - 备选：放进 `kd_sensing.utils`。拒绝原因：`utils` 会扩大轻量导入面，且更容易变成杂物间。

4. **训练 extension 框架暂不作为第一刀强删。**
   - 做法：本 change 可评估 `TrainingExtension` 是否仍是低价值接口；若删除会牵动 JEPA、teacher guidance、batch runner 和 resume metadata，任务中单独列为可选/后置收敛项，必须有 focused tests 覆盖。
   - 备选：把 extension 框架和 barrel/helper 一次性删除。拒绝原因：它触碰训练语义，风险高于其它纯表面积收敛项。

5. **护栏只读 tracked source，不扫描 ignored 本地状态。**
   - 做法：架构边界测试读取 `git ls-files`、pyproject、OpenSpec、docs 和源码文本，拒绝 tracked `__pycache__`、`.pyc`、runtime outputs、重复 facade 和禁止 import；忽略本地未跟踪 cache。
   - 备选：让测试扫描工作树全部文件。拒绝原因：本地运行产物不是源码契约，扫描它们会制造噪声。

## Risks / Trade-offs

- [Risk] 删除 package barrel 可能影响未登记的外部脚本导入。→ Mitigation：先迁移仓库内部 import；只删除没有 current spec/README/docs 公开承诺的 barrel；必要时在最终说明中标记 breaking import cleanup。
- [Risk] registry unknown-name 错误比 removed-name 错误少了迁移提示。→ Mitigation：只移除完全退役且已有 guard/tombstone 覆盖的历史名；保留有当前迁移价值的提示。
- [Risk] 合并 helper 可能把 diagnostics family 之间的细微语义抹平。→ Mitigation：只合并输出格式和错误处理一致的 helper；否则保留局部私有 helper。
- [Risk] 新增护栏过严导致正常小模块失败。→ Mitigation：检查只覆盖明确禁止的 facade/barrel/runtime artifact 模式；业务热点仍走 inventory rationale。

## Migration Plan

1. 只读扫描 tracked imports、package barrels、registry removed entries 和重复 helper。
2. 按“内部 import 迁移 → 删除/收缩包装 → 更新测试/文档 → 增加护栏”的顺序实施。
3. 每个删除项先运行对应 focused tests；跨 registry/config/import 边界后运行架构边界、配置加载和 component registry 测试。
4. 若某个删除项影响过大，回滚该项并在 inventory 标为 `merge-candidate` 或 `right-size-accepted`，不阻塞其它低风险收敛。

## Open Questions

- `TrainingExtension` 框架是否在本 change 内删除，取决于实施时的影响面；默认先不作为必须完成项。
- 是否需要新建窄 diagnostics helper 模块，取决于重复 helper 的语义是否完全一致；默认优先局部复用。
