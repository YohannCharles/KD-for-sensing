## 1. 实现前审计与边界确认

- [x] 1.1 阅读 `docs/agent_navigation.md`、`openspec/specs/project-architecture/spec.md`、本 change 的 `proposal.md`、`design.md` 和四个 delta spec，确认本次只新增 opt-in Predictive GPS-query++ 与 advantage slice，不改默认训练入口
- [x] 1.2 梳理现有 JEPA downstream、GPS-query pooler、Image ResNet+GPS baseline、difficulty operator、benchmark manifest 和 diagnostics 聚合路径，记录需要改动的最小文件集合
- [x] 1.3 确认当前 H5/G2/F1、scene32-34、future=1、seed=17、P0-P5 与 k=1..5 结果的 provenance，作为 strict comparison 的 baseline metadata

## 2. Difficulty Pipeline 与 Advantage Slice

- [x] 2.1 新增或扩展 visual-ambiguous hard negative operator，支持同 split/scene 约束、视觉相似度 proxy 或 embedding source、beam offset 下限、seed、fallback mode 和 replay metadata
- [x] 2.2 新增 beam-offset-constrained wrong GPS replacement，记录 peer sample id、beam offset、GPS distance、selection pool size、scene/split constraint、fallback count 和 warnings
- [x] 2.3 新增 combined GPS-query advantage perturbation profile，覆盖 `C3_random_async` 或 `C4_severe_async` 与 `D3_motion_blur`、`D4_partial_occlusion`、`D6_burst_missing`、`D7_joint_worst_case` 的组合
- [x] 2.4 保证 combined perturbation 对 temporal sequence 的 history source range 满足 no-future-leak，并在 replay metadata 中写出可审计范围
- [x] 2.5 为 advantage difficulty 增加 determinism 测试：同 seed 输出 tensor/mask/source index/peer id/replay metadata 完全一致，不同 seed 可改变 peer 但仍满足约束
- [x] 2.6 使用 `conda run -n kd_mm_beam pytest tests/test_modality_difficulty.py -q` 跑通 difficulty 相关最小验证

## 3. Predictive GPS-query++ 架构

- [x] 3.1 在 JEPA downstream registry/config 中新增 opt-in `pooler.type: predictive_gps_query` 或等价类型，保持 `mean`、`gps_query_attention`、`hybrid_residual_query` 语义不变
- [x] 3.2 实现 current content latent 分支，支持 learned content query 或 mean/content anchor，并保持输出 `[B,T,D]` 可被现有 projector、representation core 和 beam head 消费
- [x] 3.3 实现 GPS-conditioned residual latent 分支，使 GPS-query 只作为相对 content anchor 的 residual/bias，提供 residual scale、初始化或 gate 控制以避免 GPS path 覆盖 anchor
- [x] 3.4 实现 opt-in causal temporal latent predictor，第一版优先支持轻量 GRU，并记录 predictor type、history window、source history range、availability mask 和 insufficient-history fallback
- [x] 3.5 增加 temporal predictor no-future-leak 测试，确保 step `t` 的 predicted latent 只读取 `< t` 的 image latent，不读取 future frame、target label、beam power 或 batch sample order
- [x] 3.6 实现 reliability-aware gate，融合 current image latent、temporal predicted latent 和 GPS residual latent，并只消费连续 reliability fields、mask 与 latent consistency scores
- [x] 3.7 增加 gate 输入隔离测试，确保 `condition`、`predictive_condition_id`、`gps_condition`、`image_condition`、`c_idx`、`d_idx` 不进入 gate input tensor，diagnostics 写出 `condition_id_consumed=false`
- [x] 3.8 输出 Predictive GPS-query++ runtime diagnostics，包括三条分支 availability、gate weights、residual scale、GPS-query attention summary、temporal source history range、fallback/warning 状态
- [x] 3.9 写出架构 metadata，包括 pooler type、content query count、GPS query count、temporal predictor type、reliability gate type、residual scale、auxiliary losses、JEPA checkpoint path 和 context encoder freeze 状态
- [x] 3.10 增加 checkpoint compatibility 检查，避免把旧 `gps_query_attention` checkpoint 静默解释为 Predictive GPS-query++，除非显式 non-strict transfer
- [x] 3.11 使用 `conda run -n kd_mm_beam pytest tests/test_gps_conditioned_jepa.py tests/test_modular_sequence_next_query_transformer.py -q` 跑通 JEPA downstream 相关最小验证

## 4. 训练目标、配置与 Strict Comparison

- [x] 4.1 增加 opt-in predictive latent auxiliary loss 配置，支持 predicted/corrupt latent 与 clean detached target latent 或配置 target representation 对齐，并默认不启用
- [x] 4.2 确保未声明 auxiliary loss 时 supervised beam loss、metrics、checkpoint workflow 和 model output adaptation 与当前流程一致
- [x] 4.3 新增 H5/G2/F1、scene32-34、future=1、seed=17 的 Predictive GPS-query++ 训练配置，并明确 GPS history/source window、image history window、prediction horizon 和 label space
- [x] 4.4 新增 strict comparison manifest，至少包含 `Image ResNet+GPS`、当前 `JEPA GPS-query k=4` 和 `Predictive GPS-query++` 三组，并声明 config path、weights path、checkpoint provenance、metric profile、split、sample count、difficulty digest 和 label space
- [x] 4.5 在聚合逻辑中校验 strict comparison 字段一致性：history window、GPS input/source window、prediction horizon、scene set、seed、difficulty digest、distance metric 和 beam label space 不一致时禁止 claim upgrade
- [x] 4.6 使用 `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py tests/test_jepa_gps_shortcut_benchmark.py -q` 跑通配置加载与 benchmark manifest 相关验证

## 5. 评测、指标与诊断产物

- [x] 5.1 扩展 predictive robustness 评测，使 advantage slice 只补充 P0-P5，不替代 canonical P0-P5 主 claim 表
- [x] 5.2 输出 advantage slice per-condition DBA/Top-K 指标，并计算 Predictive GPS-query++ 相对 `Image ResNet+GPS` 与当前 `JEPA GPS-query k=4` 的 per-condition margin
- [x] 5.3 实现 GPS-query++ claim gate，分别计算 P-suite margin、advantage-slice margin vs Image ResNet+GPS、advantage-slice margin vs GPS-query k=4，并把 advantage-only 提升标为 mechanism evidence 或 partial/pending
- [x] 5.4 生成 diagnostics bundle manifest，链接 gate weight summaries、branch availability、latent consistency summaries、fallback counts、per-condition margin tables 和 explanatory figures
- [x] 5.5 增加 gate/latent/attention 可视化脚本，输出 branch weight by condition、latent consistency by condition、target rank CDF、PCA/t-SNE 或 UMAP 解释图，并在报告中注明图只作为 explanatory diagnostics
- [x] 5.6 确保所有真实 CSV/PNG/JSON/checkpoint/TensorBoard 产物写入 ignored `outputs/`、`logs/` 或 manifest 指定目录，不进入源码变更

## 6. 真实实验执行与最终验证

- [x] 6.1 使用 `conda run -n kd_mm_beam` 启动 Predictive GPS-query++ 训练，按配置记录 GPU、seed、checkpoint provenance、history window、prediction horizon 和 final config
- [x] 6.2 使用 `conda run -n kd_mm_beam` 对 `Image ResNet+GPS`、当前 `JEPA GPS-query k=4`、`Predictive GPS-query++` 跑同协议真实 P0-P5 evaluation，并输出 strict comparison table
- [x] 6.3 使用 `conda run -n kd_mm_beam` 跑 GPS-query advantage slice evaluation，输出 per-condition DBA/Top-K、fallback warnings、difficulty digest 和 claim gate summary
- [x] 6.4 使用 `conda run -n kd_mm_beam` 生成同一套可视化和 diagnostics bundle，并检查图表引用的 metrics 与 strict comparison manifest 一致
- [x] 6.5 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`，确认新增模块未破坏包边界和公共入口
- [x] 6.6 运行 `openspec validate add-predictive-gps-query-advantage --strict` 和 `openspec status --change add-predictive-gps-query-advantage`，确认 OpenSpec change 可进入 apply 阶段
- [x] 6.7 汇总最终实现说明：列出主 P0-P5 结果、advantage slice 结果、claim gate 结论、关键 diagnostics、未解决风险和复现实验命令
