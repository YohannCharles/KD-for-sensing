## 1. Missing Bucket Summary

- [x] 1.1 扩展 apples-to-apples / BC-style summary 的 pattern bucket 推导，输出 `missing_bucket_mapping.json` 并处理空 bucket warning。
- [x] 1.2 将 per-run、method mean/std、delta 与 rank markdown 增加 miss1/miss2/miss3 的 Top1、within_3 和 MAE 字段。
- [x] 1.3 添加 focused tests 覆盖 bucket mapping、avg_missing 排除 full、空 bucket warning 和排序规则。

## 2. Checkpoint Selection

- [x] 2.1 新增 `scripts/select_missing_aware_checkpoint.py`，支持 val split/lightweight subset、四种 selection rule、warning 和 selected checkpoint symlink/copy。
- [x] 2.2 添加 selection focused tests 或 smoke self-check，验证 score、summary CSV 与 symlink/copy 行为。

## 3. Funnel Matrix 与 Runner

- [x] 3.1 新增 `scripts/generate_scene31_funnel.py`，生成 main/quick/all 所需 run manifest 与 YAML，默认 root 为 `outputs/scene31_funnel_lmdb`。
- [x] 3.2 新增 `scripts/run_scene31_funnel.sh`，复用单 GPU worker 队列，支持 group、train/eval/auto-eval、overwrite、失败列表和 summary 调用。
- [x] 3.3 添加 generator/runner help 与 bash syntax focused tests。

## 4. Mild MP-DRO

- [x] 4.1 扩展 U-MaskBeamJEPA opt-in MP-DRO，支持 `lambda_dro`、full protection、protected weight 归一化和新版 `mpdro_mild_group_log.csv`。
- [x] 4.2 更新 MP-DRO focused tests，使用 `conda run -n kd_mm_beam pytest tests/test_u_mask_beam_jepa.py -q` 验证。

## 5. Funnel Summary 与结论

- [x] 5.1 新增 `scripts/summarize_scene31_funnel.py` wrapper，输出 funnel 指定文件名、checkpoint selection 合并和 conservative conclusion。
- [x] 5.2 添加 summary focused tests，验证 promotion labels、delta vs uniform、beam proximity 排序和结论文本。

## 6. OpenSpec 与回归

- [x] 6.1 运行 `openspec validate add-scene31-funnel-workflow --strict`。
- [x] 6.2 运行 `conda run -n kd_mm_beam pytest tests/test_scene31_next_round.py tests/test_u_mask_beam_jepa.py -q` 或记录无法运行原因。
