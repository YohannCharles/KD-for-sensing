## 1. AMBER Full 对齐

- [x] 1.1 新增或扩展 spatial-token image encoder，使 AMBER image branch 可配置为 ResNet34 spatial tokens，并补 registry/default import/metadata。
- [x] 1.2 更新 AMBER full core，输出 modality indicator weights 和 L2 regularization payload，并让 AMBER auxiliary loss 优先消费该 payload。
- [x] 1.3 更新 `configs/fusion/amber_full_architecture.yaml` 和 focused tests，验证 `seq_len=2`、`num_pred=1`、四模态、ResNet34 image、ResNet18 radar/LiDAR、无历史 beam token。

## 2. 文档和验证

- [x] 2.1 更新 `docs/experiment_protocols.md`、`docs/result_claims_registry.md` 和模型目录中的 AMBER/AMR 口径，保持 local/pending claim 边界。
- [x] 2.2 运行 `openspec validate align-paper-baselines-window2 --strict`。
- [x] 2.3 运行 `conda run -n kd_mm_beam pytest tests/test_amber_full_architecture.py tests/test_amr_net.py -q`。
- [x] 2.4 按需运行 `conda run -n kd_mm_beam pytest tests/test_component_registry.py tests/test_config_load_characterization.py -q` 或记录无法运行原因。
