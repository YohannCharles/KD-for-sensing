---
name: kd-update-claim
description: 从已绑定的本地证据更新 KD-for-sensing 的研究 claim、表格、provenance 注记或 claim-facing 文档。用于核对实验结论与协议边界；不得把 ignored、开发中或 claim-ineligible 产物提升为正式结论。
---

# KD Claim 更新

## 上下文

1. 读取 `AGENTS.md`、`docs/agent_navigation.md`、`docs/maintainer_context_index.yaml` 和与证据对应的 scoped context。
2. 读取相关 current spec、active change、目标 claim 文档和生成该证据的 resolved config/report metadata。目标文档不存在时不得猜测旧路径或恢复已删除的 claim 系统。
3. 新增 claim schema、paper export、指标定义或产物生命周期前，先建立或更新 OpenSpec change。

## 工作流

1. 核对 protocol id/fingerprint、split role、sample identity/order、seed、checkpoint/report SHA256、模型配置、指标定义和 `outer_test_accessed`。
2. 将 smoke、mock、historical、bounded、candidate、validation-selected、缺少 provenance 或 `claim_ineligible=true` 的结果排除在正式 claim 外。
3. 只从用户指定且可复算的本地证据更新现有 tracked 文档；不要根据 dashboard、日志摘要或单个未绑定 JSON 推断正式事实。
4. claim 依赖协议或评估范围时，同时更新对应 OpenSpec artifact；范围变化先改 spec，再改结论。
5. 本仓库没有 paper-export 公共 CLI。分析表、图和草稿留在 ignored `outputs/` 或 `logs/`，不得提交数据与 checkpoint。

所有 Python 命令使用 `kd_mm_beam`。按改动范围运行聚焦测试，并至少校验：

```bash
openspec validate --all --strict
conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q
```
