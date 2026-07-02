## 1. Core Loss

- [x] 1.1 扩展 `beam_prototype_alignment.py`：支持 `proto_target_type`、BTAPA soft target、fusion/modality 权重和 ADBA-aware auxiliary loss。
- [x] 1.2 扩展 `u_mask_beam_jepa.py`：从配置传递 BTAPA 参数，并在 diagnostics 中输出 `beam_ce_loss`、`proto_loss`、`btapa_fusion_loss`、`btapa_modality_loss`、`adba_proto_loss`、`total_loss`。

## 2. Configs

- [x] 2.1 新增 `configs/scene31/main_v3_strong_reliability_btapa.yaml`，保留旧 V3 配置不变。
- [x] 2.2 新增 tau、ADBA-aware、fusion-only 和 modality weight 消融配置。

## 3. Scripts

- [x] 3.1 新增 `scripts/smoke_test_btapa.py`，可用 `conda run -n kd_mm_beam python scripts/smoke_test_btapa.py` 验证 finite、归一化、mask 和 backward。
- [x] 3.2 新增 `scripts/run_btapa_experiments.sh`，默认串行，支持 dry-run、worker、max_parallel 和 gpu_ids。
- [x] 3.3 新增 `scripts/analyze_btapa_runs.py`，输出 comparison/delta CSV 和 Markdown，并打印关键结论。

## 4. Validation

- [x] 4.1 运行 `openspec validate add-btapa-prototype-alignment --strict`。
- [x] 4.2 运行 BTAPA smoke test。
- [x] 4.3 运行 launcher dry-run 和相关配置加载检查。
