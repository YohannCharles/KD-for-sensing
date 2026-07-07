## 1. Inspect 与设计边界

- [x] 1.1 阅读当前 PCPG/BPRR/supervised router、U-MaskBeamJEPA、prototype loss、eval matrix、上一轮 launcher/summary 和相关 tests。
- [x] 1.2 确认历史 baseline roots 只读使用，不修改旧 outputs、不提交训练产物。

## 2. 模型、loss 与 CLI 开关

- [x] 2.1 补齐 `average` fusion，并保证 missing mask、单模态和 no-NaN 行为。
- [x] 2.2 补齐 router feature ablation flags，并在 diagnostics/metadata 中记录 pattern/reliability/prototype margin 状态。
- [x] 2.3 补齐 prototype/head ablation flags：alignment、modality proto、circular/gaussian targets、`head_type=classifier` fallback。
- [x] 2.4 补齐训练 CLI 显式 flag 到 config override 的映射，默认行为不变。

## 3. Final launcher 与 summary

- [x] 3.1 新增 `scripts/launch_final_c2_ablation_v1.py`，生成 67 个 job，支持 GPU0-7、dry-run、skip/force、实验过滤、seed 参数、max_epochs、manifest 和 failed_jobs。
- [x] 3.2 新增 `scripts/summarize_final_c2_ablation_v1.py`，输出全部 CSV/Markdown 表并合并 baseline roots。

## 4. 测试与验证

- [x] 4.1 新增 `tests/test_final_c2_ablation_v1.py` 覆盖 router/prototype/fusion/soft_static/launcher/summary。
- [x] 4.2 运行 `openspec validate final-c2-ablation-v1 --strict`。
- [x] 4.3 运行 `conda run -n kd_mm_beam pytest -q tests/test_final_c2_ablation_v1.py`。
- [x] 4.4 运行 final launcher dry-run，确认 manifest 67 个 job 且 GPU/并发满足约束。
- [x] 4.5 运行用户指定 smoke test；若资源或数据阻塞，记录 fallback。
- [x] 4.6 按用户命令启动正式 final run 或给出可直接运行命令与阻塞原因。
