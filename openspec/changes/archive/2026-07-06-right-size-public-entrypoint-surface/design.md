## Context

`prune-missing-modality-mainline-surface` 已经把缺失模态主线的脚本和配置表面作为第一波收口对象。下一层风险是 package console scripts：`pyproject.toml` 中的 `kd-sensing-*` 入口一旦暴露，就会被用户和协作者自然理解为 public API。当前入口覆盖 core workflow、diagnostics、paper export、dataset audit、baseline reproduction 和若干研究/辅助命令，生命周期层级不够清晰。

本 change 只治理 public entrypoint surface。它不判断某个模型或实验路线是否科学上重要，只回答一个更窄的问题：这个命令是否应该长期作为 `kd-sensing-*` public command 暴露？

约束：

- 所有 Python 验证命令 MUST 使用 `conda run -n kd_mm_beam <command>`。
- 不新增旧命令 alias、compat wrapper 或把删除的 console script 映射到其它 workflow。
- 不删除 owner module 的核心能力，除非它只服务被删除 public entrypoint 且无 docs/spec/tests 消费。
- CLI glue 必须保持薄；真实 workflow 留在 owner module。

## Goals / Non-Goals

**Goals:**

- 为所有 `pyproject.toml` console scripts 建立 lifecycle 分类。
- 对保留 public CLI 补齐 owner、输出边界、help smoke 和 current docs/inventory 引用。
- 删除或降级不应长期暴露的 console scripts。
- 让 architecture boundary / doctor 能发现 pyproject、help smoke、inventory 和 docs 漂移。

**Non-Goals:**

- 不改变训练、评估、预处理或诊断的数学/指标语义。
- 不清理 `scripts/` 研究脚本；那已由前一波和后续报告 helper change 覆盖。
- 不新增统一 CLI 框架、不引入新依赖、不重写 argparse 风格。
- 不把 local/manual 实验 helper 升级为 package CLI。

## Decisions

### Decision 1: Public CLI 使用分类矩阵

每个 console script MUST 归入一个类别：

- `core_workflow`：训练、评估、预处理、run index 等长期基本入口。
- `current_diagnostic`：当前主线诊断或治理命令。
- `paper_export`：只读论文 artifact/export 命令。
- `baseline_reproduction`：论文或外部 baseline 复现命令。
- `local_manual`：不应暴露为 public CLI，除非 OpenSpec 明确要求。
- `internal_only`：保留 module/helper，但不声明 console script。
- `delete`：删除 entry point 和只服务它的 CLI wrapper。

理由：分类比“命令数量”更有用。少数 paper/baseline 命令可以保留 public，但必须说清楚它们不是核心训练 API。

替代方案：

- 只补 help smoke：能让测试绿，但无法解释入口为什么存在。
- 删除所有非 core CLI：太激进，会误伤当前 docs/spec 明确要求的 diagnostic/export/reproduction 入口。

### Decision 2: 保留 public CLI 必须有四个锚点

保留的 console script MUST 同时具备：

- pyproject entry point
- help smoke 或等价无副作用 smoke
- inventory/docs/OpenSpec current 引用
- owner module 和输出边界

理由：public API 不应该只存在于 `pyproject.toml` 一行里。

替代方案：允许临时 public CLI 只靠 pyproject 存在。这个方案短期省事，但会继续制造隐藏 public API。

### Decision 3: 降级优先于 wrapper

不再 public 暴露的入口默认降级为 internal-only module 或删除；不得新增旧命令 wrapper。若用户仍需功能，文档必须指向 owner module、保留的 public CLI 或明确 local/manual 命令。

理由：wrapper 会保留复杂度，且让退役 public surface 看起来仍被支持。

替代方案：保留 alias 并打印 deprecation warning。仓库当前治理已经多次拒绝旧入口兼容回流，不应重新打开这条路。

### Decision 4: CLI glue 不做 workflow

`src/kd_sensing/cli/*.py` 只做参数解析、轻量路径解析、调用 owner module 和 exit code。发现 CLI 文件复制训练 loop、评估 aggregation、report builder 或 benchmark suite 实现时，必须迁回 owner module 或删除入口。

理由：CLI 是用户门面，不是业务 owner。入口越多，越需要把实现放回稳定模块。

## Risks / Trade-offs

- 删除 console script 可能打断私人命令习惯 → 只删除无 current docs/spec/tests 契约的入口；保留替代 owner 或 current public command。
- help smoke 增加会让测试变慢 → 只要求 `--help` 或无副作用 dry-run，不启动训练、不读真实数据。
- 入口分类可能和后续配置收缩同时改 inventory → Change 1 和 Change 2 可并行实现，但合并时必须处理 `docs/project_surface_inventory.md` 冲突。
- module-only helper 误判为 public CLI → shared helper 不含 `main()`/console parser 时不作为 public entrypoint。

## Migration Plan

1. 枚举 `pyproject.toml` 的所有 `project.scripts`，生成分类表。
2. 对每个入口检查 README/docs/OpenSpec/tests/inventory 引用。
3. 标记保留、降级或删除；保留项补齐 help smoke 和 owner/output boundary。
4. 删除 pyproject 中不再 public 的 entry point，并删除只服务该 entry point 的 CLI wrapper。
5. 更新 architecture boundary / doctor，确保 pyproject、help smoke、inventory 和 docs 一致。
6. 验证 OpenSpec、CLI help、architecture boundary 和 surface doctor。

Rollback：若误删 public CLI，恢复 pyproject entry point 和原 CLI wrapper，同时补齐 lifecycle 记录；不得用新 alias 代替原入口。

## Open Questions

无阻塞问题。实现阶段若某个入口是否 public 存疑，默认保留为 public 但补齐 owner、help smoke 和删除触发条件，而不是半删半留。
