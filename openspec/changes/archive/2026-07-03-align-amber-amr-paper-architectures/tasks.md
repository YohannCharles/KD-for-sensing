## 1. AMBER 架构对齐

- [x] 1.1 移除 AMBER full core 的历史 beam token 路径和 metadata，确保 forward 只消费四模态输入。
- [x] 1.2 为 AMBER full 支持空间 token 输入、空间/时间/模态位置编码和缺失模态 key padding mask。
- [x] 1.3 将 AMBER CMA payload 改为 class-query cross-attention 语义，并更新 AMBER auxiliary loss。
- [x] 1.4 更新 AMBER full 配置，使 image/radar/LiDAR 使用 ResNet18-backed pretrained spatial-token encoder。

## 2. AMR-Net 论文公式对齐

- [x] 2.1 更新 AMR-Net 默认配置为论文输入口径：scene31 保持不变，image channel 设为 1，LiDAR/GPS snapshot 保持 216x2/2。
- [x] 2.2 将 PRE loss 改为基于 `pre_samples` 的 K 次 Monte Carlo latent sampling。
- [x] 2.3 将 CUAF 改为论文版 entropy、average pairwise KL、top-T margin 与分项 softmax 归一化。
- [x] 2.4 支持 `loss.amr.paper_objective_only=true`，避免额外叠加 fused focal 主损失。

## 3. 测试与文档口径

- [x] 3.1 更新 AMBER focused tests，覆盖无 history token、空间 token、CMA query payload 和配置 metadata。
- [x] 3.2 更新 AMR-Net focused tests，覆盖 K 次 PRE、论文版 CUAF diagnostics、AMR-only objective 和配置加载。
- [x] 3.3 更新 mainline/claim 文档中 AMBER 与 AMR-Net caveat，说明本次 paper-aligned local 状态。

## 4. 验证

- [x] 4.1 运行 `openspec validate align-amber-amr-paper-architectures --strict`。
- [x] 4.2 运行 `conda run -n kd_mm_beam pytest tests/test_amber_full_architecture.py tests/test_amr_net.py -q`。

## 5. 可变输入长度与默认长度对齐

- [x] 5.1 将 AMBER full 和 AMR-Net 默认配置输入长度改为 `seq_len=2`/`seq_length=2`，预测窗口保持 `num_pred=1`。
- [x] 5.2 让 AMR-Net forward 接受 `T>=1` 输入，并在 snapshot encoder 前做时间维聚合。
- [x] 5.3 更新 focused tests，覆盖默认 `T=2` 和非默认时间长度。
- [x] 5.4 重新运行 OpenSpec strict validation 与 AMBER/AMR focused pytest。
