## Context

上一轮变更已经让 modular forward 接收并应用 missing mask，诊断脚本也能证明 AMR/AMBER-lite 的 forward 通路可用。但正式评估链仍可能读取旧 `fresh_eval/`，且旧 summary 中 `mask_suspect=0` 在没有 `fresh_eval_maskfix/` 与 `mask_suspect.json` 的情况下不构成可信证据。

Scene31 当前可信 reference 是 `proto_randomdrop_subset_es40`。`proto_randomdrop_subset_reliability_fusion_es40` 只有 seed1/2 成功，seed3 因 CUDA illegal instruction 失败，因此只能作为 `auxiliary_candidate_only`。PatternFiLM d8 已有负向三 seed 信号，不再扩展。多场景验证需要先做 Scene31-34 的最小 quick screen，而不是全量重跑所有方法。

## Goals / Non-Goals

**Goals:**

- 生成 AMR/AMBER-lite 正式 `fresh_eval_maskfix/` 产物，并让 summary 优先读取与严格标记 suspect/excluded。
- 补齐 reliability fusion seed3 的 runner 与配置；若 seed3 后仍满足保守 criteria，再准备但不默认运行 seed4/5。
- 新增 Scene31-34 pooled/per-scene 最小验证 runner 与 summary，用于比较 natural、uniform、randomdrop subset 和 subset+reliability。
- 保持所有运行产物写入 ignored output root，不覆盖旧 checkpoint、旧 `fresh_eval/` 或其它 Scene31 roots。

**Non-Goals:**

- 不重训 AMR/AMBER-lite，不删除旧 checkpoint，不用旧 AMR/AMBER `fresh_eval/` 进入正式 ranking。
- 不继续 PatternFiLM、JTT、MVFR、MPDRO、beamsoft、condBTAPA 或 weakKD。
- 不默认运行 Scene31-34 全量 seeds、leave-one-scene-out 或 AMR/AMBER multi-scene。
- 不新增 package CLI；本轮保持 `scripts/` local/manual surface。

## Decisions

1. **maskfix 结果写入 sibling 目录而不是覆盖旧 eval。**  
   `eval_modular_lite_maskfix` 只对已存在 complete run 执行 fresh eval，输出到 `<run>/fresh_eval_maskfix/`。旧 `fresh_eval/` 作为历史产物保留，summary 可以 fallback 读取但必须把 modular-lite run 标记为 `mask_suspect=true`、`reason=no_fresh_eval_maskfix` 和 `excluded_from_official_ranking=true`。

2. **mask_suspect 判定由 eval artifact 自带并由 summary 复核。**  
   fresh eval 后写 `mask_suspect.json`，检查核心 pattern 指标完全相同、full-vs-missing logits 完全相同、`mask_applied=false`、missing_count 异常和 `maskfix_eval` 缺失。summary 不信任单个布尔列；它优先读 JSON，不存在则降级为 suspect。

3. **runner group 明确分离长跑与准备项。**  
   `reliability_seed3` 只跑失败 seed3，默认不覆盖 complete run，支持 `--overwrite-failed`。`reliability_seed45` 只作为显式 group，且不纳入默认 `all_new`。这样可以补证据，同时避免在 seed3 未证实时扩大计算。

4. **Scene31-34 先 pooled quick screen，再按 scene summary。**  
   新 runner 使用独立 root `outputs/scenes31_34_subset_reliability_lmdb`，先检查 scene 31/32/33/34 的 config/path 可用性，缺失时 warning 并记录到 summary。mode 1 使用 pooled 数据训练评估 seed1；mode 2 在 summary 中支持 per-scene metrics，不默认做 leave-one-scene-out。

5. **summary conclusion 保守表达。**  
   Scene31 combined conclusion 必须输出 trusted reference、reliability n/status/delta、PatternFiLM do-not-promote、AMR/AMBER included/excluded reason 和下一步建议。多场景 conclusion 必须报告 data/config availability、completed/missing/eval failures、pooled winner、scene stability 和是否建议扩 seed2/3。

## Risks / Trade-offs

- **真实 GPU/数据不可用会阻止长评估或训练完成** → runner、summary 和配置先完整落地，验证时记录未运行原因与可续跑命令。
- **历史 eval artifact schema 不一致** → summary reader 采用宽容解析，但缺少 maskfix 证据的 modular-lite run 一律排除 official ranking。
- **CUDA illegal instruction 可能再次出现** → seed3 runner 记录 GPU id、`CUDA_VISIBLE_DEVICES`、完整 traceback 和 failed status，不自动改模型。
- **多场景配置可能需要现有 dataset 支持之外的路径** → runner 做显式 availability check，缺 scene 时 warning，不 silent fail。

## Migration Plan

1. 补 OpenSpec delta、runner、summary、Scene31-34 脚本和 focused tests。
2. 运行 `openspec validate stabilize-scene31-maskfix-reliability-multiscene --strict`。
3. 运行相关静态/fixture tests；所有项目 Python 命令使用 `conda run -n kd_mm_beam`。
4. 若本地数据、checkpoint 和 GPU 可用，按顺序运行 modular mask diagnostics、maskfix eval、reliability seed3、Scene31 summary、Scene31-34 quick seed1 和 multi-scene summary。
5. 回滚时移除本 change 新增脚本/配置和 summary 逻辑；旧 outputs 与 checkpoint 不受影响。

## Open Questions

- 当前机器是否具备完成 AMR/AMBER-lite maskfix fresh eval、reliability seed3 和 Scene31-34 quick seed1 所需的 GPU、数据与 checkpoint。
- Scene31-34 pooled 训练是否已有直接多 scene dataset config 支持；若没有，本轮只生成最小配置/runner scaffolding 并在 availability check 中明确 warning。
