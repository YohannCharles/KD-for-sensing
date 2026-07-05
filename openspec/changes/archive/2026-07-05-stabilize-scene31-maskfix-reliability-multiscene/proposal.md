## Why

上一轮 Scene31 subset reference 与 modular missing-mask 修复已经完成，但正式结果链仍有三个缺口：AMR/AMBER-lite 还没有可信的 `fresh_eval_maskfix/` 产物和 summary 排除标记，reliability fusion 只有 seed1/2 且 seed3 失败，多场景 Scene31-34 还缺最小验证框架。现在需要补齐这些工程链路，避免把诊断结果、n=2 信号或单场景结论误读为正式 winner。

## What Changes

- 补齐 AMR/AMBER-lite maskfix fresh eval 正式产物：只加载已有 complete run 的 best checkpoint，输出到 `fresh_eval_maskfix/`，写出 `mask_suspect.json`、pattern metrics、日志和 checkpoint provenance，不重训、不覆盖旧 `fresh_eval/`。
- 修改 Scene31 subset reliability summary：AMR/AMBER-lite 优先读取 `fresh_eval_maskfix/`；缺少 maskfix 或被判 suspect 的 modular-lite run 必须标记 `excluded_from_official_ranking=true`，不得进入 official winner ranking。
- 扩展 Scene31 subset reliability runner：新增 `reliability_seed3` 与 `reliability_seed45` group；seed3 支持覆盖 failed run 并自动 fresh eval；seed4/5 只准备配置和显式 group，不进入默认 `all_new`。
- 新增 Scene31-34 subset reliability 最小验证 pipeline：提供 pooled quick seed1 runner、subset vs reliability seed1/2/3 显式 group、per-scene/pooled summary、delta 与稳定性排序，并独立写入 `outputs/scenes31_34_subset_reliability_lmdb`。
- 更新 combined conclusion：清晰区分 trusted Scene31 reference、reliability fusion 状态、PatternFiLM do-not-promote、AMR/AMBER-lite official ranking 状态和下一步建议。

## Capabilities

### New Capabilities
- `scenes31-34-subset-reliability-validation`: Scene31-34 pooled 与 per-scene 最小缺失模态验证 workflow、summary 和保守结论。

### Modified Capabilities
- `local-missing-modality-baselines`: AMR/AMBER-lite maskfix fresh eval 必须产出正式 maskfix artifacts，并以 suspect/excluded 标记控制 ranking。
- `scene31-next-round-experiment-workflow`: Scene31 subset reliability runner 与 combined summary 增加 seed3/seed45、maskfix 优先读取和保守 promotion 状态。
- `deepsense6g-scene-selection`: 多场景 Scene31-34 quick validation 必须检查 scene data/config 可用性并隔离输出 root。
- `observability-aware-fusion`: reliability fusion 的 seed 扩展和 promotion gate 必须以 randomdrop subset reference 为比较对象，且不启用其它研究线。

## Impact

- 影响脚本：`scripts/run_scene31_subset_reliability.sh`、`scripts/summarize_scene31_subset_reliability.py`、必要时新增或补齐 `scripts/run_scene31_modular_maskfix_eval.sh`、`scripts/run_scenes31_34_subset_reliability.sh` 和 `scripts/summarize_scenes31_34_subset_reliability.py`。
- 影响配置生成：Scene31 reliability seed3/4/5 配置，Scene31-34 pooled quick seed1 与 subset/reliability seed123 配置或生成逻辑。
- 影响本地运行产物：新增 ignored output roots 下的 `fresh_eval_maskfix/`、summary CSV/Markdown/TXT 和运行日志；不提交 checkpoint、数据、cache 或旧 outputs。
- 影响验证：OpenSpec validate、runner/summary focused tests、诊断脚本 smoke；真实训练和 fresh eval 取决于本地 GPU、数据和已有 checkpoint 可用性。
