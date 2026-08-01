---
name: kd-add-config
description: 在 KD-for-sensing 中新增或修改 canonical config、PCPF 本地 template、overlay、resolved config 约束和配置迁移保护。用于配置任务，并避免恢复退休路线或让 tracked template 依赖本地产物。
---

# KD 配置改动

## 上下文

1. 读取 `AGENTS.md`、`docs/agent_navigation.md`、`docs/maintainer_context_index.yaml` 和 `docs/agent_context/configs.md`。
2. 读取 `openspec/specs/u0-mainline/spec.md`、`openspec/specs/clean-data-integrity/spec.md`，以及任务相关的 active change spec。
3. 配置若改变训练流程、数据契约、兼容性或公共入口，先建立或更新 OpenSpec change。

## 工作流

1. 先区分 canonical recipe、PCPF tracked template、运行时 resolved config 和 ignored 本地产物。
2. 复用现有 `_base_` loader、严格 parser 与 owner 字段；未知字段和不完整的 stage/protocol 绑定必须失败关闭。
3. tracked template 必须在没有 `dataset/`、`outputs/`、cache 或 checkpoint 时可解析；本地路径、hash、checkpoint 和 gate 只在显式 resolve 时绑定。
4. PCPF 配置保留在 `tools/configs/pcpf/`，不得成为公共 CLI 或新增 `configs/mmw/` canonical route。
5. 不新增旧 YAML alias、兼容字段或仅靠文档实现的运行语义，也不提交生成的 resolved config 和运行产物。

所有 Python 命令使用 `kd_mm_beam`。至少运行：

```bash
conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py tests/test_pcpf_runner_protocol.py -q
make verify-cli-config
```
