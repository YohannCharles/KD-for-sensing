## 1. 契约、基线与治理边界

- [x] 1.1 复核现有 `jepa_visual_architecture_sweep` manifest、summary writer、strict comparability fields 和 follow-up 结果路径，记录当前 anchors：`patch14_stage1_gps_query`、`overlap_k16_s8_stage1`、`overlap_k20_s10`、`resnet18_layer4_tokens`、`resnet18_layer3_layer4_tokens`。
- [x] 1.2 新增 full sweep focused test 骨架，覆盖 manifest schema、candidate expansion、job dependency、runner dry-run、summary claim gate 和 retired KD guard。
- [x] 1.3 定义 runtime output root：`outputs/analysis/cnn_hybrid_jepa_visual_prior_sweep/`，确认所有 generated configs、logs、metrics、checkpoints、summaries 和 run status 都写入该 ignored 目录。
- [x] 1.4 在 tests 中锁定 runner 不修改 `/root/.container_env`、profile、SSH、systemd、dataset、tracked weights 或其他实验 output root。
- [x] 1.5 运行 `openspec validate add-cnn-hybrid-jepa-visual-prior-sweep --strict`，确认 proposal/design/spec/tasks 契约有效。

## 2. Full sweep manifest 与候选矩阵

- [x] 2.1 新增 source-managed full sweep manifest，例如 `configs/diagnostics/cnn_hybrid_jepa_visual_prior_sweep_manifest.yaml`，包含全局 split、seed、metric、output root、GPU 和并行默认值。
- [x] 2.2 扩展或新增 manifest validator，要求每个候选包含 `variant_id`、`family`、`stage_plan`、`checkpoint_policy`、`checkpoint_selection`、`availability`、token metadata、params metadata、strict comparability metadata 和 metrics path。
- [x] 2.3 定义 `existing_controls` 候选：`gps_only_control`、`patch16_mean_baseline`、`patch16_gps_query_pool`、`patch14_stage1_gps_query`、`overlap_k16_s8_stage1`、`overlap_k20_s10`、`resnet18_layer4_tokens`、`resnet18_layer3_layer4_tokens`。
- [x] 2.4 定义 `patch_resolution_stage1` 候选轴：patch size 16、14、12、10、8，默认 image size 224、GPS-query K2 frame pooler、Stage 1 + downstream。
- [x] 2.5 定义 `overlap_stage1` 候选轴：kernel/stride 12/6、14/7、16/8、20/10、24/12，默认 GPS-query K2 frame pooler、Stage 1 + downstream。
- [x] 2.6 定义 `cnn_supervised_tokens` 候选轴：backbone ResNet18/ResNet34，token source layer3/layer4/layer3+layer4，pretrained scratch/ImageNet，freeze full_ft/freeze_backbone_projection/unfreeze_layer4。
- [x] 2.7 定义 `cnn_jepa_tokens` 候选轴：backbone ResNet18/ResNet34，token source layer4/layer3+layer4，pretrained scratch/ImageNet，Stage 1 + downstream。
- [x] 2.8 定义 `hybrid_tokenizers` 候选轴：conv_stem_s16、conv_stem_s8、local_patch16、local_patch14、cvt_patch16、cvt_patch14、conv_stem_patch14、conv_stem_patch12。
- [x] 2.9 定义 `pooler_core_ablation` 候选轴：mean、gps_query_k1_frame、gps_query_k2_frame、gps_query_k4_frame、gps_query_k2_tokens、gps_query_k4_tokens、hybrid_residual_query、token_aware_core。
- [x] 2.10 定义 `teacher_guided_stabilization` 候选轴：teachers 为 resnet18_l4_scratch、resnet18_l4_imagenet_unfreeze_layer4、resnet34_l4_imagenet；students 为 patch14、patch12、overlap_k16_s8、conv_stem_s16、local_patch14；temperature 为 2/4；weight 为 0.1/0.3/0.5。
- [x] 2.11 定义 `compute_controls` 候选轴：matched_trainable_params、matched_token_count、frozen_backbone_controls、random_feature_controls，并记录匹配目标和控制变量。
- [x] 2.12 定义 `seed_confirm` 展开规则，full mode 默认对所有 primary candidates 展开 seeds 17、23、42，并记录 seed aggregation policy。
- [x] 2.13 为暂未实现的候选保留 manifest 行并标记 `availability: requires_component`，生成器不得静默丢弃全量候选。

## 3. 配置与 job manifest 生成器

- [x] 3.1 新增 config/job generator，支持 `--mode full`、`--dry-run`、`--output-root`、`--families`、`--force` 和 `--skip-unavailable`。
- [x] 3.2 生成 `generated_configs/pretraining/*.yaml`，覆盖 patch resolution、overlap、CNN JEPA tokens 和 hybrid tokenizers 的 Stage 1 configs。
- [x] 3.3 生成 `generated_configs/downstream/*.yaml`，覆盖 supervised-only anchors、Stage 1 downstream、pooler/core ablation、CNN token downstream 和 compute controls。
- [x] 3.4 生成 teacher-guided student configs，必须使用当前 `loss.teacher_guidance`，记录 teacher checkpoint/logits source、temperature、weight 和 stop-gradient policy。
- [x] 3.5 生成 `generated_configs/eval/*.yaml`，覆盖 primary、best_top1 和可选 last checkpoint selection 的 re-evaluation。
- [x] 3.6 生成 `jobs/stage1.tsv`、`jobs/downstream.tsv`、`jobs/teacher_guided.tsv`、`jobs/reeval.tsv`、`jobs/summary.tsv` 和 `jobs/all.tsv`，每行包含 `job_id`、`variant_id`、`stage`、`depends_on`、`command`、`output_dir`、`metrics_path`、`log_path`。
- [x] 3.7 所有生成的项目 Python 命令必须使用 `conda run -n kd_mm_beam`，并把 `CUDA_VISIBLE_DEVICES` 留给 runner 注入。
- [x] 3.8 生成 `manifest_expanded.json` 和 `manifest_expanded.csv`，保留 full matrix 中 skipped、failed、missing 和 unavailable 候选。
- [x] 3.9 添加 generator dry-run 测试，验证 full mode 候选族完整、job 依赖无环、Stage 1 jobs 先于 downstream jobs。

## 4. 模型组件与配置缺口补齐

- [x] 4.1 检查并补齐 patch tokenizer 对 patch size 16、14、12、10、8 的支持，确保 token grid、token count、position interpolation 和 token budget metadata 正确。
- [x] 4.2 检查并补齐 overlap tokenizer 对 kernel/stride 12/6、14/7、16/8、20/10、24/12 的支持，添加 shape/token metadata focused tests。
- [x] 4.3 检查并补齐 CNN feature-map token source 对 ResNet18/ResNet34、layer3/layer4/layer3+layer4、scratch/ImageNet、freeze policies 的配置化支持。
- [x] 4.4 检查并补齐 CNN token source 作为 JEPA Stage 1 visual encoder 的 opt-in 路径，确保输出 `[B,T,N,D]` 和 `VisualTokenMetadata`。
- [x] 4.5 检查并补齐 hybrid tokenizers：conv stem、local token mixing、CvT-like convolutional projection、conv_stem_patch14、conv_stem_patch12。
- [x] 4.6 检查并补齐 pooler/core ablation：mean、GPS-query K1/K2/K4 frame、GPS-query K2/K4 token output、hybrid residual query、token-aware core。
- [x] 4.7 新增 params/compute inspector，记录 total params、trainable params、visual params、pooler params、token count、attention token proxy 和 backbone family。
- [x] 4.8 新增 teacher-guidance config adapter，只允许 `loss.teacher_guidance`；添加测试确保 retired `distillation` key、旧 whole-model KD route 和 legacy teacher config 会失败。
- [x] 4.9 为所有新增或复用组件添加 synthetic forward tests，覆盖 batch shape、metadata、checkpoint policy 和不兼容配置报错。

## 5. Full sweep runner 与运行安全

- [x] 5.1 新增 source-managed thin runner 或生成 `run_full_sweep.sh`，默认读取 `jobs/all.tsv` 并在 output root 下写入 logs/status。
- [x] 5.2 runner 默认 GPU 列表为 `0,1,2,3`，默认 `max_parallel=8`，并在每个 job 运行前注入 `CUDA_VISIBLE_DEVICES=<gpu>`。
- [x] 5.3 runner 必须尊重 `depends_on`，只有依赖 job 成功或已存在 success marker 时才启动下游 job。
- [x] 5.4 runner 支持 `--dry-run` 输出完整命令清单，不启动训练。
- [x] 5.5 runner 支持 `--resume`，当 metrics path 和 success marker 存在时默认跳过该 job 并记录 skip reason。
- [x] 5.6 runner 支持 `--retry-failed` 和 `--force-rerun`，并将重跑原因写入 status table。
- [x] 5.7 runner 支持 `--clean-output-root`，且只能删除 `outputs/analysis/cnn_hybrid_jepa_visual_prior_sweep/` 或用户显式传入的 sweep output root。
- [x] 5.8 runner 写出 `run_status.jsonl`、`run_status.csv` 和当前并发快照，便于用户确认为什么少于 8 个任务在跑。
- [x] 5.9 添加 shellcheck 或最小 bash syntax 检查，并用 dry-run fixture 测试 GPU 分配、并行上限、依赖阻塞和 cleanup 范围。

## 6. Summary、Pareto 与结果解释

- [x] 6.1 新增或扩展 summary aggregator，读取 expanded manifest、job status、training metrics、evaluation metrics 和 diagnostics。
- [x] 6.2 输出 `summary/full_results.csv`、`summary/full_results.json`、`summary/strict_ranking.csv`、`summary/family_best.csv`、`summary/pareto.csv` 和 `summary/eval_summary.md`。
- [x] 6.3 full results 每行必须保留 missing、failed、skipped、unavailable 状态，不得只汇总成功结果。
- [x] 6.4 strict ranking 只允许 `strict_comparable: true`、非 smoke-only、checkpoint provenance 完整、metrics 存在且 split/seed/metric 一致的候选进入。
- [x] 6.5 family best 汇总每个 family 的 Top-1、Top-3、Top-5、DBA、beam distance、checkpoint selection、seed mean/std 和相对 anchors 差值。
- [x] 6.6 Pareto 汇总按 DBA、Top-1、trainable params、total params、token count、compute proxy 生成候选，并标记 CNN/ImageNet/frozen/teacher-guided/hybrid 来源。
- [x] 6.7 checkpoint-selection 汇总对比 primary、best_top1 和 last，避免不同 checkpoint selection 的结果覆盖。
- [x] 6.8 seed aggregation 汇总 seeds 17、23、42 的 per-seed、mean、std、best 和稳定性排名。
- [x] 6.9 Markdown summary 必须给出“分辨率 vs overlap vs CNN 局部先验 vs 预训练/冻结 vs teacher-guided”的可解释表格，而不是只输出最终第一名。

## 7. 测试覆盖

- [x] 7.1 添加 manifest schema tests，覆盖必填字段、候选族完整性、availability、checkpoint policy、stage plan 和 strict comparability metadata。
- [x] 7.2 添加 candidate expansion tests，验证 patch、overlap、CNN supervised、CNN JEPA、hybrid、pooler、teacher-guided、compute controls 和 seeds 全部展开。
- [x] 7.3 添加 config generation tests，使用 `conda run -n kd_mm_beam pytest ... -q` 验证生成 YAML 可被现有 config loader 解析。
- [x] 7.4 添加 job dependency tests，验证 Stage 1 -> downstream、teacher -> student、downstream -> re-eval、all -> summary 的依赖无环且可拓扑排序。
- [x] 7.5 添加 runner dry-run tests，验证 GPU 0-3、最多 8 进程、resume/skip、retry-failed 和 cleanup 范围。
- [x] 7.6 添加 summary/claim gate tests，覆盖 missing metrics、smoke-only、seed mismatch、checkpoint mismatch、split mismatch 和 unavailable rows。
- [x] 7.7 添加 teacher-guidance guard tests，确保 `loss.teacher_guidance` 可用且 retired `distillation` key 或旧 KD route 会失败。
- [x] 7.8 添加 architecture boundary tests，确认未新增 root-level legacy scripts、未绕过 `src/kd_sensing` 包结构、未提交 runtime outputs。

## 8. 验证命令

- [x] 8.1 运行 `openspec validate add-cnn-hybrid-jepa-visual-prior-sweep --strict`。
- [ ] 8.2 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [x] 8.3 运行 `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py -q`。
- [x] 8.4 运行新增 full sweep focused tests，例如 `conda run -n kd_mm_beam pytest tests/test_cnn_hybrid_jepa_visual_prior_sweep.py -q`。
- [x] 8.5 运行相关 JEPA/visual diagnostics focused tests，例如 `conda run -n kd_mm_beam pytest tests/test_gps_conditioned_jepa.py tests/test_modality_visual_diagnostics.py -q`。
- [x] 8.6 运行 generator dry-run：`conda run -n kd_mm_beam python -m kd_sensing.diagnostics.cnn_hybrid_jepa_visual_prior_sweep --mode full --dry-run --output-root outputs/analysis/cnn_hybrid_jepa_visual_prior_sweep`。
- [x] 8.7 运行 runner dry-run 或 bash syntax check，确认完整命令使用 GPU 0-3、最多 8 并行且不启动训练。
- [x] 8.8 运行 summary fixture test，确认 full table、strict ranking、family best、Pareto 和 Markdown summary 都能从 synthetic metrics 生成。

## 9. 全量运行交付

- [x] 9.1 生成完整 full sweep runtime bundle：expanded manifest、generated configs、job TSV、run script、summary script 和 README-like run note。
- [x] 9.2 用 dry-run 输出完整 sh 命令清单，人工检查候选数量、Stage 1/downstream/teacher-guided/re-eval 依赖和日志路径。
- [x] 9.3 在用户确认需要实际启动时，运行 full runner，默认 GPU 0-3、最多 8 个项目进程并行。
- [x] 9.4 运行期间使用 `run_status.csv` 或 runner status 命令解释当前并发数量、等待依赖、失败和已完成任务。
- [ ] 9.5 训练完成后运行 summary aggregator，生成 `outputs/analysis/cnn_hybrid_jepa_visual_prior_sweep/summary/eval_summary.md`。
- [ ] 9.6 根据 strict ranking、family best、Pareto 和 seed stability 给出下一步架构选择建议：继续提高分辨率、改用 CNN/hybrid、使用 ImageNet 冻结/微调、还是采用 teacher-guided stabilization。
