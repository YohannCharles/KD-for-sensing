## Context

当前入口分为 package CLI、thin CLI alias、research diagnostic、dataset preparation 和 shell orchestration。这个分类已经在 inventory 和 maintainer context index 中存在，但还缺少“CLI 文件能写什么、不能写什么”的结构性规则。

## Goals / Non-Goals

**Goals:**

- 固化 parser/IO 与真实实现的职责边界。
- 为 maintainer context index 的 entrypoint 增加 owner module 和 output boundary。
- 防止新增脚本复制训练循环、评估循环、dataset 读取主逻辑或 retired route compatibility wrapper。
- 保留当前用户可用入口。

**Non-Goals:**

- 不删除当前 CLI。
- 不重命名 console scripts。
- 不把所有 research diagnostic 立即迁移到 package CLI。
- 不改变 argparse 参数或输出。

## Decisions

### Decision 1: CLI 文件只做薄 orchestration

package CLI 可做 argparse、配置加载、override 解析、调用包内 API 和打印/写出 summary；不得实现大段训练循环、评估循环、模型 forward 或 dataset parsing。

### Decision 2: entrypoint index 增加 owner metadata

每个 entrypoint 建议记录：

- `owner_module`: 真实实现模块。
- `responsibility`: parser、thin alias、diagnostic、dataset preparation 等。
- `output_boundary`: ignored output root 或 no-write/read-only。
- `retired_route_guard`: 是否需要显式防回流。

### Decision 3: 架构测试用轻量启发式检查

测试不需要理解所有业务逻辑，但可以限制 CLI 文件行数、禁止明显训练 loop marker、要求 owner module 存在，并要求 scripts allowlist entry 有职责和输出边界。

## Risks / Trade-offs

- [Risk] 某些 CLI 需要较多参数映射，看起来不够“薄”。  
  → Mitigation: 允许 parser/override glue，但要求真实 workflow 在 owner module。

- [Risk] research diagnostic 脚本被误判。  
  → Mitigation: 按 lifecycle 分类使用不同预算和 marker；不是所有脚本都要求同样薄。

- [Risk] 增加 index 字段维护负担。  
  → Mitigation: 字段短小，换来新增入口时更清晰的 review 面。
