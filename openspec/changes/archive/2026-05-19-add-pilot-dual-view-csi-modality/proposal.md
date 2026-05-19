## Why

项目当前支持 image、radar、GPS、LiDAR 和 mmWave power vector，但还不能把原始复数 CSI 作为可训练的一等模态接入 beam prediction。`CSI编码器.md` 已经给出更符合通信系统叙事的 noisy CSI 建模方式：输入不是直接加 AWGN 的 CSI，而是上行导频相关后的 channel estimate `h_hat = h + e`，适合用于构造可解释的弱化 CSI 模态和 SNR/pilot 消融。

## What Changes

- 新增 CSI 数据能力：从序列 CSV 读取 `csi1..csiN` 历史路径列，加载复数 CSI 张量或 real/imag 张量，并提供训练集全局 RMS 统计供编码器固定归一化使用。
- 新增 pilot-based noisy CSI estimator：支持物理参数模式 `noise_var / (pilot_power * pilot_len)` 和实验 SNR 模式，训练时可在 `[train_snr_min_db, train_snr_max_db]` 内采样，评估时使用固定 `snr_db`。
- 新增 dual-view CSI encoder：先得到 noisy channel estimate，再生成 frequency view 与 delay/IFFT view，经两个 CNN tokenizer、`mean`/`concat`/`symmetric_gate` 融合和 1 层 GRU 输出 `[B, T, 64]` 特征。
- 将 `csi` 纳入中心化模态契约、batch 准备、模型注册、`modular_sequence` 编码器配置、fusion `modalities` 和默认组件导入。
- 新增 CSI-only teacher/student/no-KD 或统一 modular 配置，并新增 CSI 与现有多模态融合配置示例，优先支持 `modular_sequence` 路径。
- 新增测试覆盖 CSI 张量加载、RMS 统计复用、pilot noise 方差、dual-view shape、注册表构建、batch future padding、单模态训练 smoke 和 fusion 输入对齐。

## Capabilities

### New Capabilities
- `csi-channel-data`: 定义 CSI 序列列、复数张量加载、训练集 RMS 统计、future padding 和 clean label 对齐契约。
- `csi-modality-model`: 定义 pilot-based noisy dual-view CSI encoder、CSI-only beam prediction 配置、SNR/pilot 消融参数和模型输出契约。

### Modified Capabilities
- `modality-contracts`: 将 `csi` 纳入受支持模态顺序、样本字段、fusion 输入字段、dataset flag、默认字段和标准化校验。
- `modality-aware-data-loading`: 将 `csi` 纳入启用模态推导、按模态懒加载、batch 准备和单模态任务输入路径。
- `configurable-multimodal-fusion`: 将 `csi` 纳入 fusion 可选模态，使 modular fusion 能接收 CSI encoder 输出并与其它模态对齐。
- `component-registry`: 要求 CSI encoder/model 通过现有注册表发现、构建，并由默认组件导入路径注册。

## Impact

- 影响代码：`src/kd_sensing/modalities.py`、`src/kd_sensing/data/datasets/`、`src/kd_sensing/data/transform_ops/`、`src/kd_sensing/engine/batch.py`、`src/kd_sensing/engine/modality_resolution.py`、`src/kd_sensing/models/`、`src/kd_sensing/models/modular.py`、`src/kd_sensing/registries.py`、`configs/`、`tests/`。
- 数据接口：启用 CSI 的 split CSV 需要提供历史 `csi1..csiN` 列；beam label 仍来自 clean beam power 或既有 `future_beam*` 标签，不使用 noisy CSI 重新生成标签。
- 配置接口：新增 `experiment.task: csi`、`data.dataset.use_csi`、`model.*.encoders.csi.type: pilot_dual_view_csi` 和 `csi_estimation`/`dual_view` 参数。
- 兼容性：旧五模态配置和数据路径必须继续可加载；没有启用 CSI 时不得要求 `csi*` 列或读取 CSI 文件。
- 依赖影响：优先使用 NumPy/PyTorch 处理复数张量、IFFT 和噪声采样，不新增外部依赖。
