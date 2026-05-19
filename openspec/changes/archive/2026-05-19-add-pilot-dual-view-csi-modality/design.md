## Context

项目当前通过 `src/kd_sensing/modalities.py` 维护中心化模态契约，训练与评估路径再由 `modality_resolution`、`engine/batch.py`、`modular_sequence` 和注册表串联。现有一等模态为 `image`、`radar`、`gps`、`lidar`、`mmwave`，其中 mmWave 使用 64 维 beam power vector；MMW Town10 准备流程已经保留 channel path 并从 channel 派生 beam power，但模型层还没有原始复数 CSI 输入。

`CSI编码器.md` 的关键约束是：noisy CSI 不应建模为直接对输入 CSI 加 AWGN，而应建模为上行导频相关后的估计误差。编码器顺序必须是 clean CSI → 训练集全局 RMS 归一化 → pilot-based channel estimate `h_hat = h + e` → frequency/delay 双视图 → CNN tokenizer → 视图融合 → GRU → `[B, T, 64]`，并且标签仍由 clean beam power 或既有 future beam label 生成。

## Goals / Non-Goals

**Goals:**

- 将 `csi` 加入项目的一等模态合同，使单模态和 fusion 配置能通过同一批数据、batch 和模型路径运行。
- 提供可配置、可测试的 `pilot_dual_view_csi` encoder，支持物理参数模式和 estimation SNR 模式。
- 保持 CSI encoder 输出 `[B, T, D]`，使其能直接作为 `modular_sequence` encoder 接入现有 projector、representation core、head、KD 和 G2D 路径。
- 为 CSI 数据读取提供明确输入格式：复数 tensor 或 last-dim real/imag tensor，并复用训练集 RMS 统计而不是 per-sample RMS。
- 保持旧五模态数据和配置兼容；未启用 CSI 时不得要求 CSI 列或读取 channel 文件。

**Non-Goals:**

- 不在本次变更中重写 beam label 生成逻辑；noisy CSI 不参与重新生成标签。
- 不引入新的跨模态融合架构；CSI 优先接入现有 `modular_sequence`、`early_concat_gru` 和 token transformer 路径。
- 不要求一次性支持所有外部 channel 文件格式；实现应先覆盖本项目准备流程可产出的 `.npy`/`.npz`/real-imag tensor，并对不支持字段给出清晰错误。
- 不把完整 pilot 接收矩阵 `Y` 作为默认训练数据保存；默认使用正交 pilot correlation 后的等价噪声模型。

## Decisions

### Decision 1: `csi` 作为第六个一等模态

将 `MODALITY_ORDER` 扩展为 `("image", "radar", "gps", "lidar", "mmwave", "csi")`，新增 `dataset_flag="use_csi"`、`sample_keys=("csi",)`、`fusion_input_key="csi_batch"` 和模型默认字段。这样 dataset、batch、fusion 和诊断能沿用既有模态合同模式。

替代方案是把 CSI 归入 mmWave 的附属字段。该方案会混淆 beam power vector 和复数 channel tensor：前者是 64 维观测功率，后者是 `[Nsc, Nant, 2]` 或 complex channel，归一化、噪声模型和消融含义都不同。

### Decision 2: 数据层读取 clean CSI，噪声在模型层注入

Dataset 返回 clean CSI 历史张量，encoder 内部根据训练/评估配置生成 `h_hat`。这样同一份数据能服务 clean、0/5/10/20/30 dB 和 pilot 参数消融，且标签始终来自 clean beam power。

替代方案是在预处理阶段写出多个 noisy CSI cache。该方案会膨胀数据占用，也容易把随机训练 SNR 和评估固定 SNR 固化到数据文件里。

### Decision 3: 使用训练集全局 RMS 统计

CSI 归一化使用训练 split 全局 RMS，并作为 normalizer artifact 复用到 test split。encoder 不做 per-sample RMS normalization，因为 per-sample normalization 会削弱 pilot SNR 的物理含义。

替代方案是 batch 内动态归一化。该方案实现简单，但会让不同 SNR/pilot 参数之间的有效噪声强度不可比。

### Decision 4: 默认实现等价 pilot estimation noise

默认不显式构造 `Y = h s^H + N`，而是在正交 pilot 假设下直接采样 `e ~ CN(0, sigma_p^2 / (pilot_power * pilot_len))` 或按 `est_snr_db` 采样 `sigma_e2`。这与文档公式等价，计算代价低，也便于按 batch 采样训练 SNR。

替代方案是构造完整 pilot 序列和接收矩阵。该方案更贴近链路仿真，但对当前 beam prediction 训练没有额外收益，且会扩大输入维度和内存占用。

### Decision 5: CSI encoder 注册为 `ENCODERS`，单模态模型优先使用 `modular_sequence`

新增 `kd_sensing.models.csi`，注册 `pilot_dual_view_csi` 到 `ENCODERS`，并可选注册 `csi_feature_extractor` 到 `MODELS`。CSI-only 配置使用 `type: modular_sequence`、`modalities: [csi]`、`encoders.csi.type: pilot_dual_view_csi`，避免新增一套 teacher/student 输出契约。

替代方案是新增 `csi_teacher` 和 `csi_student` 独立模型。该方案与 mmWave 历史做法一致，但会重复 GRU/head/KD 代码；如后续需要轻量 CSI student，可在同一模块里补注册名。

### Decision 6: 视图融合默认 `symmetric_gate`

frequency view 和 delay view 使用同构但不共享参数的 CNN tokenizer，默认通过 symmetric gate 融合，同时支持 `mean` 和 `concat` 做消融。Delay view 必须从 noisy estimate `h_hat` 做 IFFT，不能先 IFFT 再分别加噪。

替代方案是只用 frequency view。该方案可作为消融配置，但默认方案应保留 delay-domain 结构以体现多径稀疏性。

## Risks / Trade-offs

- [Risk] 不同数据源的 channel 文件 shape 不一致 → Mitigation：先定义 dataset 输出规范 `[T, Nsc, Nant, 2]` 或 complex `[T, Nsc, Nant]`，loader 对不支持 shape 报错，并在 MMW/DeepVerse 适配层做显式转换。
- [Risk] CSI 张量比现有向量模态更大，DataLoader 可能变慢 → Mitigation：支持 `.npy/.npz` 直接 mmap/按样本读取，优先只加载启用模态，并保持未启用 CSI 时零开销。
- [Risk] 训练集 RMS 统计需要扫描训练 split → Mitigation：仿照 mmWave/LiDAR normalizer artifact，只在 train dataset fit，一次保存并复用；测试集不得重新 fit。
- [Risk] SNR 配置语义混乱 → Mitigation：配置区分 `mode: physical` 与 `mode: est_snr`，日志和 aux diagnostics 记录 `sigma_e2`、`snr_db`、`pilot_len` 和 `pilot_power`。
- [Risk] fusion 组合数量随第六模态上升 → Mitigation：本次只要求配置系统能接收 `csi` 和提供代表性 CSI-only/CSI-fusion 示例，不强制生成所有 57 个多模态组合实体 YAML。
