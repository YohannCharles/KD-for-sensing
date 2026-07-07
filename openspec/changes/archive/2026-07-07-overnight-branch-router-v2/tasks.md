## 1. Inspect 与定位

- [x] 1.1 定位主训练入口、Scene31-34 训练/eval 入口、TinyViT 配置、PCPG/BPRR/raw/oracle gate、hard subset、JEPA、branch/radar auxiliary、selection metric、drop-count eval、summary/launcher 和现有 tests。
- [x] 1.2 确认上一轮 `outputs/pcpg_radar_balance_v1` 与 `outputs/bprr_reliability_router_v1_retry_gpus0_6_20260706_193654` 的可读 artifact 形态，不修改旧 outputs。

## 2. 训练与 Router 实现

- [x] 2.1 实现 `soft_static` hard subset weighting helper、pattern alias 和 run config/summary 字段，默认行为保持不变。
- [x] 2.2 实现 supervised router helper：oracle target、masked softmax、focus pattern 判断、单模态处理和 pattern_best fallback。
- [x] 2.3 将 `--fusion supervised_router`、router supervision/distill/focus/fuse-level 参数接入现有训练/PCPG/BPRR 路径，并保证 branch aux、radar protect、hard subset、JEPA、selection metric 可组合或 fail fast。
- [x] 2.4 保存 router diagnostics、oracle target distribution 和 router accuracy，且 oracle 诊断不混入真实 ranking。

## 3. Launcher 与 Summary

- [x] 3.1 新增 `scripts/launch_overnight_branch_router_v2.py`，生成 A/B/C 共 40 个 job，支持 GPU1-2 并发、dry-run、skip_completed、force、seed/experiments/max_epochs/output_root/baseline_root 等参数。
- [x] 3.2 新增 `scripts/summarize_overnight_branch_router_v2.py`，聚合当前 root 与 baseline roots，写出 summary/drop-count/pattern/router CSV 和 summary.md 自动结论。

## 4. 测试与验证

- [x] 4.1 新增 `tests/test_overnight_branch_router_v2.py` 覆盖 soft_static、oracle target、masked softmax、focus pattern、launcher dry-run 和 summary parser。
- [x] 4.2 运行 `openspec validate overnight-branch-router-v2 --strict`。
- [x] 4.3 运行 `conda run -n kd_mm_beam pytest -q tests/test_overnight_branch_router_v2.py`。
- [x] 4.4 运行 launcher dry-run，并确认 manifest 只使用 GPU1/2、总任务数 40。
- [x] 4.5 运行 smoke test：按用户本轮要求使用 GPU4-6，`conda run -n kd_mm_beam python scripts/launch_overnight_branch_router_v2.py --gpus 4,5,6 --max_jobs 3 --per_gpu 1 --anchor_seeds 1 --explore_seeds 1 --experiments b3,c1 --output_root outputs/overnight_branch_router_v2_smoke --max_epochs 1 --force`。
- [x] 4.6 smoke 通过后，按用户本轮要求使用 GPU4-6；因每卡 2 个 TinyViT 训练实测 OOM，改用稳定并发 `conda run -n kd_mm_beam python scripts/launch_overnight_branch_router_v2.py --gpus 4,5,6 --max_jobs 3 --per_gpu 1 --anchor_seeds 1,2,3,4,5 --explore_seeds 1,2,3 --output_root outputs/overnight_branch_router_v2 --skip_completed` 启动正式 overnight。
