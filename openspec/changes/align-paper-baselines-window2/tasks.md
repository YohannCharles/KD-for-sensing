## 1. AMBER Full 对齐

- [x] 1.1 新增或扩展 spatial-token image encoder，使 AMBER image branch 可配置为 ResNet34 spatial tokens，并补 registry/default import/metadata。
- [x] 1.2 更新 AMBER full core，输出 modality indicator weights 和 L2 regularization payload，并让 AMBER auxiliary loss 优先消费该 payload。
- [x] 1.3 更新 `configs/fusion/amber_full_architecture.yaml` 和 focused tests，验证 `seq_len=2`、`num_pred=1`、四模态、ResNet34 image、ResNet18 radar/LiDAR、无历史 beam token。

## 2. RMBP-MM 对齐

- [x] 2.1 新增 `rmbp_channel_attention_fusion` representation core，覆盖 AvgPool/MaxPool/shared-MLP/sigmoid/missing-mask 逻辑、metadata 和 synthetic forward。
- [x] 2.2 更新 RMBP-MM workflow helper，提供 random available modality masking 与 similarity-based modality imputation 的 batch augmentation，并保护 target/sample metadata。
- [x] 2.3 更新 WCL/RMBP-MM local substitute config、source-audit metadata 和 tests，使默认模态为 `image/radar/gps/lidar`、`seq_len=2`、`num_pred=1`，并移除 `mmwave` 输入。

## 3. 文档和验证

- [x] 3.1 更新 `docs/experiment_protocols.md`、`docs/result_claims_registry.md` 和模型目录中的 AMBER/AMR/RMBP-MM 口径，保持 local/pending claim 边界。
- [x] 3.2 运行 `openspec validate align-paper-baselines-window2 --strict`。
- [x] 3.3 运行 `conda run -n kd_mm_beam pytest tests/test_amber_full_architecture.py tests/test_amr_net.py tests/test_wcl2025_missing_modality.py -q`。
- [x] 3.4 按需运行 `conda run -n kd_mm_beam pytest tests/test_component_registry.py tests/test_config_load_characterization.py -q` 或记录无法运行原因。
