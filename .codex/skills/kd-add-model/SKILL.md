---
name: kd-add-model
description: 在 KD-for-sensing 中新增或修改模型组件、baseline、registry entry、forward/loss 契约和模型侧配置。用于四模态 topology predictor 或保留 baseline 的模型工作，并保持 OpenSpec、注册表、训练流程和本地产物边界。
---

# KD 模型改动

## 上下文

1. 读取 `AGENTS.md`、`docs/agent_navigation.md`、`docs/maintainer_context_index.yaml` 和 `docs/agent_context/models.md`。
2. 按维护索引读取 `openspec/specs/u0-mainline/spec.md` 与任务相关的 active change spec；数据或产物契约受影响时再读取对应 current spec。
3. 非平凡功能、架构、训练流程、数据契约、兼容性或公共入口改动必须先建立或更新 OpenSpec change。

## 工作流

1. 先确认现有 owner、registry 和组件能否承载改动；沿用 `src/kd_sensing` 的当前模块边界。
2. 模型、loss 与本地实验工具各归其 owner；不得复制通用 trainer，也不得增加兼容聚合层或退休入口。
3. 明确 forward shape、可训练参数、checkpoint metadata 和 config parser 失败条件。
4. 只新增覆盖实际契约的聚焦测试；模型公共边界变化同时运行 registry 与架构测试。
5. 不提交 `dataset/`、`outputs/`、`logs/`、cache、TensorBoard 或新 checkpoint。

所有 Python 命令使用 `kd_mm_beam`。聚焦验证从维护索引列出的 model tests 开始，并按风险补充：

```bash
conda run -n kd_mm_beam pytest tests/test_component_registry.py tests/test_architecture_boundaries.py -q
make verify-compile
```
