# KD for Sensing

PCPF-T 是当前唯一 active research mainline，研究共享 beam prototype、逐模态时序建模、拓扑风险与解析可靠性融合。U0、AMBER-Full、RMBP-MM、DeepSense6G Scene31--34 T2、MMW trajectory baseline 与 CSI/TSPC 继续作为稳定基线或后续研究基础保留。

MMW 训练只能通过经审计的 clean-inner 或 trajectory-disjoint protocol 启动。outer test、confirmation train 和任何要求隔离的资源重叠都会在创建数据 loader 前被拒绝。

## PCPF-T 主线

PCPF-T 使用独立注册模型 `pcpf_temporal_risk_fusion`，本地配置位于 `tools/configs/pcpf/`。它复用共享 trainer、MMW 数据协议和现有四模态 encoder，不是第四个 canonical MMW recipe，也不会把 PCPF 参数加入 U0。

当前 PCPF 开发结果固定为 `claim_ineligible: true`，outer test 保持封存。实现与实验任务以 `openspec/changes/add-pcpf-temporal-risk-fusion/` 为权威。

```bash
conda run -n kd_mm_beam python tools/run_pcpf.py resolve --stage stage1 --output outputs/pcpf_temporal_risk/resolved/stage1.yaml
conda run -n kd_mm_beam python tools/run_pcpf.py preflight --config outputs/pcpf_temporal_risk/resolved/stage1.yaml
conda run -n kd_mm_beam python tools/run_pcpf.py synthetic-smoke
conda run -n kd_mm_beam python tools/run_pcpf.py one-batch-smoke
```

Stage 2/3 的 `resolve` 还必须传入前一阶段 validation-best checkpoint；Stage 3 还必须传入未截断且通过的 gate report。A0--A4 与分量消融通过 `--template tools/configs/pcpf/ablations/<recipe>.yaml` 选择。正式训练只由显式的 `tools/run_pcpf.py train --config <resolved.yaml>` 启动。

## MMW 工作流

所有项目命令使用 `kd_mm_beam` 环境。先从本地 split manifest 生成 protocol 和审计报告：

```bash
conda run -n kd_mm_beam python scripts/audit_clean_inner_protocol.py \
  --source-manifest /path/to/inner_split_manifest.json \
  --protocol-output outputs/mmw_clean_u0/protocol.yaml \
  --audit-json outputs/mmw_clean_u0/audit.json \
  --audit-md outputs/mmw_clean_u0/audit.md

conda run -n kd_mm_beam python scripts/launch_mmw_all_weather_matrix.py \
  --protocol outputs/mmw_clean_u0/protocol.yaml \
  --audit-report outputs/mmw_clean_u0/audit.json \
  --output-root outputs/mmw_clean_u0 \
  --methods U0 --seeds 1 --gpus 0

conda run -n kd_mm_beam python scripts/eval_mmw_all_weather_matrix.py \
  --root outputs/mmw_clean_u0 --methods U0 --seeds 1
```

`configs/mmw/u0.yaml`、`amber_full.yaml` 和 `rmbp_mm.yaml` 是 tracked canonical recipe；它们不携带本地 MMW split，不能绕过 protocol 直接训练。`mmw_trajectory_disjoint_v1`、M0--M4 与因果消融保留为本地研究工作流。

## CSI / TSPC

CSI、稀疏导频、Radio、TSPC/TSPC-V2 及其 trajectory/full-pool 依赖继续保留。它们必须使用过去帧输入、train-only codebook/cache、封存 test，并保持 Full 与 CSI-off 硬旁路；这些入口不扩展 public CLI 或 canonical recipe。

## DeepSense6G

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/deepsense6g/t2.yaml
conda run -n kd_mm_beam kd-sensing-evaluate --help
conda run -n kd_mm_beam kd-sensing-preprocess --help
```

DeepSense6G 保持自己的 Scene31--34、四模态、64 类 future-beam split 契约，不会被 MMW protocol 重解释。

`dataset/`、`outputs/`、`outputs/cache/`、`cache/`、日志和 checkpoint 都是本地产物，源码收敛不会移动或删除它们。

## 验证

```bash
make verify-quick
make verify-cli-config
make verify-compile
conda run -n kd_mm_beam pytest -q
```
