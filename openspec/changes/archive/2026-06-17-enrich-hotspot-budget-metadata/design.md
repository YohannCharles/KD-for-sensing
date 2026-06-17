## Context

当前 hotspot governance 能阻止超长函数/类静默扩大，但缺少“行动元数据”。inventory 中有推荐拆分方向和暂缓原因，但它是自然语言，测试和 Codex 不能可靠地读取优先级、目标模块或下一步 change。

## Goals / Non-Goals

**Goals:**

- 为 hotspot budget 增加机器可读行动字段。
- 让 Codex 能直接从 index 判断 P0/P1/P2、目标模块、验证命令和推荐 change。
- 保持 inventory 的解释和 caveat 角色。
- 不改变已有 line budget 检查语义。

**Non-Goals:**

- 不在本 change 中拆热点代码。
- 不把所有 inventory 长说明复制到 YAML。
- 不新增外部 schema 依赖。

## Decisions

### Decision 1: 给每个 budget 增加最小行动字段

建议字段：

- `priority`: `P0`、`P1`、`P2`、`P3`。
- `status`: `split-next`、`monitor`、`facade-budget`、`defer-with-rationale`。
- `split_targets`: 目标模块或 helper 名称列表。
- `next_change`: 推荐 OpenSpec change 名称或空值。
- `rationale`: 一句短理由，详细说明仍在 inventory。
- `validation_commands`: focused tests 列表。

### Decision 2: metadata 校验只检查结构，不判断技术正确性

测试验证字段存在、值合法、路径存在、命令使用 `kd_mm_beam`，不试图判断拆分目标是否最佳。技术判断留给 OpenSpec design 和 review。

### Decision 3: inventory 保持长解释

index 的 `rationale` 是短摘要，inventory 继续记录“为什么暂缓”和“为什么这样拆”。两者冲突视为治理漂移。

## Risks / Trade-offs

- [Risk] YAML 变长。  
  → Mitigation: 只给 budget entries 加短字段，不复制长段解释。

- [Risk] priority 变成主观标签。  
  → Mitigation: 允许 review 调整；测试只验证合法值。

- [Risk] next_change 过期。  
  → Mitigation: status 可设为 `monitor` 或更新为 archive/current 状态，架构测试只检查格式。
