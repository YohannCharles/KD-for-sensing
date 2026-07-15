## 1. 资源切换与本地审计

- [x] 1.1 按用户决定停止 GPU0-3 的 DeepSense S1/T2 seeds2/3 和旧评估编排器，确认 GPU4-7 的 LG/CLS seed1 PID、tmux 与显存未受影响。
- [x] 1.2 只读审计 3 天气 × 5 场景 manifest、H5/P1 split、metadata/sanity、availability、cache、radar source 和四 baseline 输入边界，记录当前阻断。

## 2. Strict split 与 readiness

- [x] 2.1 修正 P1 future-label reuse eligibility：继续输出 reuse diagnostics，但不将单步类别重复单独判为结构性泄漏。
- [x] 2.2 让 MMW availability/preflight 识别 `metadata_h5p1.json`、`sanity_report_h5p1.json` 和显式 split tag，并 fail closed 校验 strict metadata。
- [x] 2.3 使用 `conda run -n kd_mm_beam pytest tests/test_mmw_town10_preparation.py -q` 覆盖 P1 重复标签、真实结构重叠和 H5/P1 readiness。

## 3. 四传感器数据准备

- [x] 3.1 新增最小 all-weather radar/preflight 配置，复用公开 `mmw_radar_maps` 和 `mmw_sequence_splits_from_manifest`，不在 dataset loader 中静默写 CSV。
- [x] 3.2 串行为 15 个 domain 生成 radar RA/DA maps、materialized split CSV 与 metadata，保持原 H5/P1 和 zip 不变。
- [x] 3.3 用 `conda run -n kd_mm_beam` 运行 15-domain preflight 和单样本 image/GPS/LiDAR/radar shape smoke，确认 sensitive fields 未进入 sensing input。

## 4. Pooled domain runtime

- [x] 4.1 实现 `data.dataset.domains` 的显式 MMW leaf dataset + `ConcatDataset` 构建和 domain provenance，拒绝缺字段/重复 id/缺 split。
- [x] 4.2 使用 PyTorch `WeightedRandomSampler` 实现 opt-in domain-balanced train sampling，并记录 seed、replacement、num_samples 和每 domain 权重；validation/test 保持全量遍历。
- [x] 4.3 使用 `conda run -n kd_mm_beam pytest` 增加 focused tests，覆盖 15-domain 构建、等 domain 权重、可复现与非 train split 禁用 sampler。

## 5. 四方法实验入口与评估

- [x] 5.1 新增 local/manual MMW all-weather launcher，生成 S1、T2、AMBER-Full、RMBP-MM seed1 配置；固定 H5/P1、四 sensing modalities、missing augmentation、epoch/optimizer budget、last-checkpoint policy 和 GPU0-3 一卡一进程。
- [x] 5.2 新增固定 whole-modality 15-subset 与 temporal 0/20/40/60/80% evaluator，严格记录 domain/sample/mask/cache/geometry/sensitive-usage provenance。
- [x] 5.3 新增 per-domain、weather macro、scene macro、15-domain macro、micro、worst-domain、paired delta 和 reliability diagnostics summary；local validation 与正式 claim eligibility 分开标记。

## 6. 验证与 dry-run

- [x] 6.1 运行 `openspec validate run-mmw-all-weather-missing-modality-matrix --strict` 和 `openspec validate --all --strict`。
- [x] 6.2 运行 `conda run -n kd_mm_beam` focused tests、`make verify-quick`、`make verify-cli-config` 和 `make verify-compile`。
- [x] 6.3 运行四方法 dry-run 与每方法 1-batch smoke，确认 GPU0-3 映射、run root、config、log、checkpoint 和失败状态互不覆盖。

## 7. GPU0-3 seed1 validation screening

- [x] 7.1 在 GPU0-3 并行完成四方法 fixed-epoch seed1 训练；不使用 validation 指标早停或选择 best checkpoint，统一冻结 `last.pth`。
- [x] 7.2 使用共享 v2 missing cache 重新评估四方法，生成 weather/scene/domain summary、mask 间不确定性和 S1-T2 paired evidence。
- [x] 7.3 根据 rainy/foggy macro、worst-domain、clean guardrail、missing robustness 和 calibration 决定直接保留 T2，或为明确失败模式另提 reliability module change。

## 8. temporal 评估修复与 baseline 论文审计

- [x] 8.1 对照 `paper/rmbp_mm.pdf` 与 `paper/AMBER.pdf`，记录输入模态、时序建模、训练阶段、缺失构造、损失、encoder 和评估协议与本地实现的差异。
- [x] 8.2 将 temporal cache 改为 v2 几何覆盖协议：0% 单 mask、modality-frame 多个互异 mask、frame/block 全支持枚举，并 fail closed 拒绝旧三 mask cache。
- [x] 8.3 在 raw metrics 和 summary 中增加实际缺失率、末帧可用性、trailing missing、mask 间均值/标准差/最差值与 baseline reproduction scope。
- [x] 8.4 增加 focused tests，运行 OpenSpec strict validation、相关 pytest 和 compile verification。
- [x] 8.5 在可用 GPU 上并行重跑 v2 evaluator；RMBP temporal 标记为 out-of-paper-scope diagnostic，不把其曲线用于论文等价结论。

## 9. 85/90/95% 极端稀疏评估

- [x] 9.1 为现有 evaluator 增加可选 rate、mask type 和跳过 whole-modality 的窄参数，保证 85/90/95% 分别精确保留 3/2/1 个 cells，并增加 focused tests。
- [x] 9.2 在不干扰 GPU4-7 保留任务的前提下，于 GPU0-3 并行运行四方法、15-domain、16 fixed masks/rate 的 extreme modality-frame evaluation。
- [x] 9.3 合并现有 80% modality-frame 与新 85/90/95% 结果，输出均值、标准差、最差 mask、天气/域分层和 T2-S1 paired delta。

## 10. 融合表征稳定性诊断

- [x] 10.1 新增 local/manual 配对融合表征脚本与 focused tests，统一提取 beam-head 输入特征，复用 v2 `modality_frame` masks，并实现方法内 clean PCA、原空间漂移与邻近 beam 指标。
- [x] 10.2 在 GPU0-7 分片提取 T2、AMBER-Full、RMBP-MM 的 15-domain clean/20/40/60/80% 表征，校验 checkpoint、sample、domain 与 mask provenance 后合并。
- [x] 10.3 输出 clean PCA、missing shift PCA、rate-level mask mean/std/worst、per-domain CSV 和结论边界；仅在 matched T2 no-alignment ablation 完成后才允许将差异归因于 Beam Prototype Alignment Loss。

## 11. 高维循环拓扑论文图

- [x] 11.1 扩展现有 local/manual 表征脚本与 focused tests，增加无新依赖的 cosine Gram/profile、固定 kNN-Isomap、phase/Topo 指标和 signed circular feature-shift 汇总。
- [x] 11.2 直接消费已提取的 15-domain 64 维产物，生成 T2 prototype 三联图、三方法 clean centroid Isomap 对比、20/40/60/80% feature-shift 热图及 CSV/JSON/Markdown provenance，不重跑模型。
- [x] 11.3 运行 focused tests、compile 与 OpenSpec strict validation，并检查图片非空、分辨率、统一色标、数值一致性和 no-BPA 因果边界。

## 收口结果

- 四方法 extreme evaluation 各输出 720 行，覆盖 15 domain、85/90/95% 三档和每档 16 个固定 `modality_frame` mask；合并 summary 已包含 80/85/90/95% 分层结果。
- rainy 与 foggy 的 T2-S1 temporal Top1 delta 均高于门禁，最终决策为 `keep_t2_no_weather_module`；结果仍是 fixed-epoch seed1 local validation，不升级为正式多 seed claim。
