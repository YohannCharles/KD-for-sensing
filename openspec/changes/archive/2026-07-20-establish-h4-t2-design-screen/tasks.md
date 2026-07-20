## 1. H4 Profile 与 Legacy 隔离

- [x] 1.1 在 MMW all-weather builder 中实现 `umask_h4_v1` / `legacy_h0_v1` 显式 profile，禁止 builder 在 profile 后静默覆盖 optimizer、weight decay 或 scheduler。
- [x] 1.2 为 T2/S1 mainline、T2 BPA/CMA ablation 和 legacy H0 hyperparameter screen 分别接入正确 profile，并将 profile canonical values 与 SHA256 写入 protocol provenance。
- [x] 1.3 使用 `conda run -n kd_mm_beam pytest tests/test_mmw_all_weather_runtime.py tests/test_mmw_t2_hyperparameter_screening.py tests/test_config_load_characterization.py -q` 验证 H4、H0、S1 和 baseline 的实际 resolved config。

## 2. 受控 T2 结构候选

- [x] 2.1 扩展 U-MaskBeamJEPA，使 `reliability_mean` fusion 与默认 supervised router 共存，并保持 router/oracle loss 和 metadata 契约可审计。
- [x] 2.2 实现 mask-aware `masked_attention` temporal pooling，保留默认 masked-mean 输出与严格的有效 temporal cell 校验。
- [x] 2.3 为 GPS MLP 增加仅训练期、默认关闭的 normalized-feature jitter 配置，并记录 effective GPS encoder/noise metadata。
- [x] 2.4 使用 `conda run -n kd_mm_beam pytest tests/test_u_mask_beam_jepa.py tests/test_s1_temporal_superset_training.py tests/test_component_registry.py -q` 覆盖默认兼容、mask、梯度和配置失败路径。

## 3. 分阶段 Design-Screen 运行面

- [x] 3.1 新增独立 MMW T2 design-screening launcher，复用 group-safe inner split、40-epoch `last.pth`、共同 batch probe 与 manifest 完整性校验。
- [x] 3.2 实现第一波八个单因素 H4 候选：control、`d_model=48/96`、router hidden `32/128`、router pattern feature off、GPS MLP hidden `32/128`。
- [x] 3.3 实现后续结构、BPA、CMA/KL 候选的 matched-control/allowlist/provenance 规则，保持 BPA/CMA 互斥与 development-only 边界。
- [x] 3.4 使用 `conda run -n kd_mm_beam pytest tests/test_mmw_t2_design_screening.py tests/test_mmw_all_weather_runtime.py -q` 验证 generated config、fingerprint、inner split、profile 和选择门槛。
- [x] 3.5 对 development-only design config 禁用 final test，确保 outer test loader 不构造、不迭代且 `final_test_metrics` 只记录未执行状态。
- [x] 3.6 增加 outer-test isolation regression，并重新运行 H4 generated-config、fingerprint、inner split 和 profile focused tests。
- [x] 3.7 使 design config/candidate recipe fingerprint 忽略 CLI 注入的 transient `runtime` metadata，并覆盖 CLI 启动后的 provenance 校验。

## 4. 验证与第一波执行

- [x] 4.1 使用 `conda run -n kd_mm_beam` 完成 H4/legacy config dry-run、8 个候选单 optimizer-step smoke 和相关 OpenSpec 校验。
- [x] 4.2 在 GPU0--7 上完成实际 step batch probe，冻结所有成功卡共用的 16 倍数 batch 和 probe manifest。
- [x] 4.3 在 GPU0--7 并行启动第一波八个 seed1、40-epoch H4 design-screen 训练，并记录每个 run 的 profile、candidate、GPU、inner split 与完成状态。此前 `outputs/mmw_t2_design_screening_v1` 使用补丁前 YAML 并消费了 outer test，仅保留为失效审计产物；有效首波已在 `outputs/mmw_t2_design_capacity_v3` 使用显式 `training.final_test.enabled=false` 完成。
- [ ] 4.4 训练完成后仅按 inner validation 的预注册保护门槛筛选最多两个候选；不消费 outer test。
- [x] 4.5 将 RouterNoPattern 固化为 T2/S1 共用的开发 mainline router architecture profile，保持 tracked base、legacy H0 与 BPA/CMA ablation 的 pattern-on 配置不变，并使其 checkpoint/evaluation/summary provenance fail closed。

## 5. 后续波与收尾

- [ ] 5.1 对晋级候选补 seed2/3，并在独立 outer evidence protocol 中确认 H4 主线，不把 development 结果升级为论文 claim。
- [ ] 5.2 在当前 BPA/CMA formal change 收口后运行 BPA 单因素与 NoBPA-CMA/KL 后续波，并输出 matched-control 分析。
- [x] 5.3 使用 `conda run -n kd_mm_beam make verify-quick`、`conda run -n kd_mm_beam python scripts/verify_compile.py`、`openspec validate establish-h4-t2-design-screen --strict` 和 `openspec validate --all --strict` 进行回归，确认 outputs 未进入源码变更。
