## 1. 上下文与现有实现审计

- [x] 1.1 阅读 modular forward、batch kwargs 过滤、fresh eval missing mask、baseline pack runner/summary、PatternFiLM 和 reliability 相关实现，确认最小复用点。
- [x] 1.2 搜索 reliability/quality/gate/mask_weight/weighted_fusion 等关键词，生成 `outputs/scene31_subset_reliability_lmdb/reliability_fusion_audit.md` 或可复现的审计脚本。

## 2. Modular missing mask 修复与诊断

- [x] 2.1 修改 `ModularSequenceModel.forward` 签名，显式接受 `missing_mask`、`missing_modality_metadata`、`available_modalities` 和 `modality_mask` 默认参数。
- [x] 2.2 在 modular model fusion/core 前应用 availability mask，确保缺失模态不贡献 fused feature，并保留 bounded debug diagnostics。
- [x] 2.3 新增 `scripts/diagnose_modular_missing_mask.py`，输出 forward 签名、batch 过滤、mask 应用、full-vs-missing logits equality 和 diagnosis CSV。
- [x] 2.4 增加 synthetic smoke/focused tests，覆盖 missing mask 影响 logits、metadata 不丢失和 silent identical-output warning。

## 3. Maskfix fresh eval runner

- [x] 3.1 新增或扩展 `scripts/run_scene31_baseline_pack_maskfix_eval.sh`，只加载 complete run 的 best checkpoint 重跑 AMR/AMBER-lite fresh eval。
- [x] 3.2 新增 `scripts/run_scene31_subset_reliability.sh --group eval_modular_lite_maskfix`，复用 maskfix eval，支持多 GPU、跳过、overwrite-eval、日志和失败列表。
- [x] 3.3 在 eval 后自动标记 `mask_suspect=true`，并输出要求的 pattern-wise metrics 字段与 checkpoint provenance。

## 4. Subset reference 与 summary

- [x] 4.1 新增 `scripts/summarize_scene31_subset_reference.py`，读取 baseline pack fresh eval，优先使用 `proto_randomdrop_subset_es40` n>=3 实际值，fallback 时 warning。
- [x] 4.2 输出 per-run、method mean/std、delta vs subset、rank markdown、suspect modular results 和 conservative conclusion。
- [x] 4.3 增加 summary fixture tests，确认 uniform 降级为 ablation、mask_suspect 排除 ranking、delta 全部相对 subset。

## 5. Reliability fusion 与 subset/PatternFiLM 配置

- [x] 5.1 若已有 reliability fusion 可复用，则接入 proto/randomdrop subset 配置；否则新增最小 mask-weighted reliability fusion，缺失模态 weight=0，可用模态归一化。
- [x] 5.2 增加 reliability weight epoch 聚合日志 `reliability_weights_epoch.csv`，字段包含 epoch、pattern、modality、mean_weight、std_weight、available_rate。
- [x] 5.3 新增或生成 `proto_randomdrop_subset_reliability_fusion_es40_seed1/2/3` 配置，确保禁用 condBTAPA、weakKD、MPDRO、beamsoft、AMBER。
- [x] 5.4 新增或生成 `proto_randomdrop_subset_pattern_film_d8_es40_seed1/2/3` 配置，确保 randomdrop subset exposure、d8、identity init、pre_head 和 best checkpoint fresh eval。
- [x] 5.5 扩展 `scripts/run_scene31_subset_reliability.sh` 的 `reliability`、`subset_film` 和 `all_new` groups，支持 train/eval/auto-eval/overwrite 语义。

## 6. Combined summary 与文档台账

- [x] 6.1 新增 `scripts/summarize_scene31_subset_reliability.py`，同时读取 baseline pack、maskfix modular、新 reliability 和 subset PatternFiLM 结果。
- [x] 6.2 输出 combined per-run、method mean/std、delta vs subset、rank markdown、suspect modular results 和 promotion conclusion。
- [x] 6.3 按主线变化同步必要文档台账：`docs/mainline_model_catalog.md`、`docs/experiment_protocols.md`、`docs/result_claims_registry.md`、`docs/experiment_matrix.md`，必要时补 `docs/mainline_experiment_history.md`。

## 7. 验证与运行

- [x] 7.1 运行 `openspec validate fix-scene31-subset-reference-mask-eval --strict`。
- [x] 7.2 运行相关 focused tests，所有 Python 命令使用 `conda run -n kd_mm_beam`，至少覆盖 modular mask、summary fixture、runner dry-run/静态检查和架构边界。
- [x] 7.3 如本地数据、checkpoint 和 GPU 可用，运行诊断、AMR/AMBER-lite maskfix fresh eval、reliability fusion、subset PatternFiLM d8 和 combined summary；无法完成长训时记录未运行原因和可续跑命令。
  - 本会话已运行 modular missing-mask 诊断、reliability audit、配置生成和 summary/runner focused tests；未启动 AMR/AMBER-lite 完整 maskfix fresh eval、reliability 三 seed 训练或 subset PatternFiLM d8 三 seed 训练，避免占用长 GPU 队列。续跑命令已记录在 `docs/experiment_matrix.md`。
