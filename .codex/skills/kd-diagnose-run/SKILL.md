---
name: kd-diagnose-run
description: 只读诊断 KD-for-sensing 的本地训练、评估、PCPF 产物、配置、协议 provenance 和仓库表面。用于排查运行失败、检查实验健康度、核对 ignored 产物或定位当前 CLI/config 漂移；除非用户另有明确授权，不修改本地产物或受保护文件。
---

# KD 运行诊断

## 上下文

1. 读取 `AGENTS.md`、`docs/agent_navigation.md` 和 `docs/maintainer_context_index.yaml`。
2. 根据任务路由只读取 `docs/agent_context/` 中对应的一份 context，并读取索引列出的 current spec 或 active change。
3. 诊断若暴露新的公共工作流、配置契约、指标 schema 或产物生命周期需求，先建立或更新 OpenSpec change，再实施。

## 工作流

1. 默认保持只读。先记录命令、进程、配置、协议 id/fingerprint、checkpoint/report hash 和错误原文，再判断根因。
2. PCPF-T 使用 `tools/run_pcpf.py` 与 `tools/eval_pcpf.py`；先运行对应 `--help`，再选择 `preflight`、gate 或 matrix 等现有动作。不要注册新的 console script。
3. 公共入口只允许 `kd-sensing-train`、`kd-sensing-evaluate` 和 `kd-sensing-preprocess`。旧入口缺失属于预期，不得恢复兼容层。
4. 将 train、只读 validation 与封存 outer test 明确区分。不得把 validation/test 证据误写为训练证据，也不得因诊断扫描 outer test。
5. `dataset/`、`outputs/`、`logs/`、cache 和 checkpoint 是本地产物；可以按用户请求读取，但不得提交、移动、删除或改写。清理请求先给出精确清单，删除仍需用户明确确认。
6. 只有已通过协议、provenance 和对应验证的证据才能进入正式 claim；其余结果保持 `claim_ineligible` 或诊断状态。

## 命令

所有 Python 命令使用 `kd_mm_beam`：

```bash
conda run -n kd_mm_beam python tools/run_pcpf.py --help
conda run -n kd_mm_beam python tools/eval_pcpf.py --help
conda run -n kd_mm_beam pytest tests/test_pcpf_workflow.py tests/test_pcpf_runner_protocol.py -q
make verify-quick
```

窄诊断只运行相关测试；涉及公共 CLI、导入边界或完整回归时按 `AGENTS.md` 使用 `make verify-cli-config`、`make verify-compile` 或 `make verify-full`。
