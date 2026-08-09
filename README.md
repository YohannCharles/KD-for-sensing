# KD for Sensing

当前研究主线由两部分组成：原生四模态拓扑预测器，以及缺失模态下由 sensing posterior 引导的局部波束探测。模型只接收 `image/radar/gps/lidar` 五帧历史，单阶段输出 64-beam posterior；均值、方差、熵和 MAP 均由 posterior 无参数计算。TBCP probing 使用 train-only ULA-DFT topology likelihood 与 validation-only requested-beam simulator，不读取未来 channel、完整 power vector 或标签来选择候选。

MMW 训练只允许使用经审计的 `mmw_id_stratified_block_v1`：seed 0、block size 32、70/15/15，train/validation/test 严格隔离。开发阶段固定 `claim_ineligible: true` 且 test 封存。

## 四模态拓扑预测器

权威契约位于 `openspec/specs/four-modal-topology-predictor/`，当前 probing change 位于 `openspec/changes/add-uncertainty-adaptive-local-probing/`。唯一配置族为：

- `tools/configs/topology_predictor/topology_on.yaml`
- `tools/configs/topology_predictor/topology_off.yaml`

正式运行先把 split 与 ULA-DFT audit 绑定到 resolved config，再执行 preflight 和单阶段训练：

```bash
conda run -n kd_mm_beam python tools/run_topology_predictor.py resolve \
  --template tools/configs/topology_predictor/topology_on.yaml \
  --manifest outputs/splits/mmw_id_stratified_block_v1/seed_0.json \
  --topology-audit outputs/cache/mmw_codebook_topology/v1/a692c2b43365b483/topology_manifest.json \
  --train-seed 1 --run-name topology_on_seed1 \
  --output outputs/four_modal_topology_predictor/resolved/topology_on_seed1.yaml

conda run -n kd_mm_beam python tools/run_topology_predictor.py preflight \
  --config outputs/four_modal_topology_predictor/resolved/topology_on_seed1.yaml

conda run -n kd_mm_beam python tools/run_topology_predictor.py train \
  --config outputs/four_modal_topology_predictor/resolved/topology_on_seed1.yaml
```

三 seed 的 topology-on/off 必须从头训练，不加载旧五模态或分阶段 checkpoint。

## 15-mask 与 TBCP 评估

```bash
conda run -n kd_mm_beam python tools/eval_topology_predictor.py matrix \
  --config <resolved.yaml> --checkpoint <validation-best.pth> --output <matrix.json>

conda run -n kd_mm_beam python tools/eval_topology_predictor.py fit-probe-likelihood \
  --config <resolved.yaml> --output outputs/tbcp7_probe_calibration/topology_likelihood.npz

conda run -n kd_mm_beam python tools/eval_topology_predictor.py probe-diagnostic \
  --config <resolved.yaml> --checkpoint <validation-best.pth> --matrix-report <matrix.json> \
  --topology-likelihood outputs/tbcp7_probe_calibration/topology_likelihood.npz \
  --output-dir <probe-output>
```

Matrix 固定覆盖四模态的 15 个非空 mask。主 probing 预算固定 K=7；TBCP-7、Batch-TBCP-3+4、Posterior Top-7、Local-7 与 Uniform-7 使用相同验证身份和 requested-only 测量边界。

## 其他保留路线

U0、AMBER-Full、RMBP-MM 与 DeepSense6G Scene31--34 继续使用公共 CLI：

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/mmw/u0.yaml
conda run -n kd_mm_beam kd-sensing-evaluate --help
conda run -n kd_mm_beam kd-sensing-preprocess --help
```

`dataset/`、`outputs/`、`outputs/cache/`、日志与 checkpoint 都是本地产物，不进入源码提交。

## 验证

```bash
make verify-quick
make verify-cli-config
make verify-compile
conda run -n kd_mm_beam pytest -q
```
