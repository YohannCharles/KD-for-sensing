# KD for Sensing

PCPF-T 是当前唯一 active research mainline，研究共享 beam prototype、逐模态时序建模、拓扑风险与解析可靠性融合。U0、AMBER-Full、RMBP-MM 与 DeepSense6G Scene31--34 T2 作为稳定 recipe 保留。

MMW 训练只能通过经审计的 `mmw_id_stratified_block_v1` protocol 启动。trajectory 固定为 `(scene_id,cav_id)`，三种天气按基础时间点绑定，以 128 个基础帧的连续 block 做确定性标签平衡 70/15/15 分配。当前 seed 0 为 90/19/19 blocks、31,602/6,723/6,855 windows；三个 split 均覆盖 5 个场景和全部 16 条轨迹，block/base frame/raw frame 跨 split 重叠均为 0。

## PCPF-T 主线

PCPF-T 使用独立注册模型 `pcpf_temporal_risk_fusion`，本地配置位于 `tools/configs/pcpf/`。它复用共享 trainer、MMW 数据协议和现有四模态 encoder，不是第四个 canonical MMW recipe，也不会把 PCPF 参数加入 U0。

当前 PCPF 开发结果固定为 `claim_ineligible: true`；MMW test 保持封存，全部开发评估只使用 validation。实现与实验任务以 `openspec/changes/add-pcpf-temporal-risk-fusion/` 为权威。

```bash
conda run -n kd_mm_beam python tools/run_pcpf.py prepare-trajectory
conda run -n kd_mm_beam python tools/run_pcpf.py cache-sparse-csi --split-seed 0
conda run -n kd_mm_beam python tools/run_pcpf.py resolve --stage stage1 --output outputs/pcpf_temporal_risk/resolved/stage1.yaml
conda run -n kd_mm_beam python tools/run_pcpf.py preflight --config outputs/pcpf_temporal_risk/resolved/stage1.yaml
conda run -n kd_mm_beam python tools/run_pcpf.py synthetic-smoke
conda run -n kd_mm_beam python tools/run_pcpf.py one-batch-smoke
```

Stage 2/3 的 `resolve` 还必须传入前一阶段 validation-best checkpoint；Stage 3 还必须传入未截断且通过的 gate report。A0--A4 与分量消融通过 `--template tools/configs/pcpf/ablations/<recipe>.yaml` 选择。正式训练只由显式的 `tools/run_pcpf.py train --config <resolved.yaml>` 启动。

## MMW 工作流

所有项目命令使用 `kd_mm_beam` 环境。首次使用或 source index 变化后生成 seed 0 manifest、审计、统计报告、train-only normalization 和开发集缓存：

```bash
conda run -n kd_mm_beam python tools/run_pcpf.py prepare-trajectory \
  --dataset-root dataset/MMW \
  --output-root outputs \
  --split-seed 0

conda run -n kd_mm_beam python tools/run_pcpf.py cache-sparse-csi --split-seed 0

conda run -n kd_mm_beam python scripts/launch_mmw_all_weather_matrix.py \
  --split-seed 0 --train-seeds 0 --methods U0 --dry-run
```

canonical manifest 位于 `outputs/splits/mmw_id_stratified_block_v1/seed_0.json`，报告位于 `outputs/split_reports/mmw_id_stratified_block_seed0.{md,json}`。同一 seed 和 source 重复准备会完整校验并复用；只有显式 `--regenerate` 才重写。`configs/mmw/u0.yaml`、`amber_full.yaml` 和 `rmbp_mm.yaml` 共享该 manifest；`split_seed` 与 `train_seed` 分离。公共 train/evaluate CLI 默认只构建 train/validation，只有显式 `--evaluate-test` 才能加载 test。

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
