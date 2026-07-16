# Agent Context 路由

这里为 MMW T2/baseline 与受限 DeepSense6G T2 提供按任务加载的短上下文。权威仍是 `AGENTS.md`、README、current OpenSpec、active change 与 `docs/project_surface_inventory.md`。

先读导航和维护索引；再按任务读取一个 context；最后执行索引中对应的最小验证。

| Route id | Scoped context | 适用任务 |
| --- | --- | --- |
| `model` | `models.md` | T2、S1、AMBER-Full、RMBP-MM 的模型和 loss |
| `data` | `data.md` | MMW 与 DeepSense6G 四模态数据、batch、预处理边界 |
| `config` | `configs.md` | `configs/mmw/`、`configs/deepsense6g/` recipe 与解析 |
| `cli` | `cli.md` | 三个 package CLI 与保留的 MMW helpers |
| `runtime` | `diagnostics.md` | 训练、评估、fixed-mask 和多 seed 证据 |
| `openspec` | `openspec.md` | current spec、active change 和归档 |
| `documentation` | `documentation.md` | 导航、inventory 和历史说明 |
| `claims` | `claims.md` | MMW evidence 与跨数据集 claim 边界 |
