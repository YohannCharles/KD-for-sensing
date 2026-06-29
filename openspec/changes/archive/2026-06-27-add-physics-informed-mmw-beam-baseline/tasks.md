## 1. 数据监督与 shape inspection

- [x] 1.1 新增 MMW physics supervision adapter 窄模块，复用 `MMWDataset` sample 字段生成 `physics_targets`、valid mask、field mapping metadata 和 unavailable reasons。
- [x] 1.2 支持 path 参数标准化字段 `aod`、`aoa`、`delay`、`gain_real`、`gain_imag`、`path_mask`，并实现按 gain magnitude 排序的第一版对齐 helper。
- [x] 1.3 为 CSI、beamspace power、path params、metadata 和缺失字段路径添加 synthetic adapter tests，不读取真实 `dataset/`。
- [x] 1.4 新增包内 dataset inspection/debug shape summary 能力，输出 image/CSI/beam/path/subcarrier/antenna/beam 数和 modality availability，不新增 `scripts/*.py` thin alias。

## 2. 可微物理模块

- [x] 2.1 新增 `kd_sensing.models.physics.complex_utils`，实现 real/imag 与 complex 互转、complex MSE、abs-square、angle normalization。
- [x] 2.2 新增 ULA array response helper，支持 antenna 数、carrier frequency、wavelength、spacing ratio 和 angle unit。
- [x] 2.3 新增 channel synthesizer，按预测 path 参数、subcarrier grid 和 path mask 生成 complex `h_hat`。
- [x] 2.4 新增 beam/codebook scoring helper，优先复用现有 beam codebook/DFT helper，缺真实 codebook 时记录 fallback source。
- [x] 2.5 添加 complex autograd focused tests，覆盖 finite forward、mask 行为和 `backward()` gradient。

## 3. PINN 模型与 registry

- [x] 3.1 新增 `pinn_multimodal_beam` 模型文件，实现可选模态编码、fusion latent、path head、direct head、physics head 和 hybrid logits。
- [x] 3.2 复用现有 image/GPS/LiDAR/radar/mmwave/CSI encoder 或最小 MLP/CNN fallback；不复制 dataset 解析或训练循环。
- [x] 3.3 将模型注册到 `MODELS`，加入默认组件导入，并保证 `import kd_sensing.registries` 仍保持轻量。
- [x] 3.4 实现模型输出 dict，主键 `logits` 兼容 `adapt_model_output`，diagnostics 包含 `direct_logits`、`physics_logits`、`h_hat`、`path_hat`、`latent` 和 shape metadata。
- [x] 3.5 添加 registry build、synthetic forward、missing modality 和 `adapt_model_output` focused tests。

## 4. Loss、metrics 与 metadata

- [x] 4.1 新增 physics-informed loss bundle，组合 beam CE、beam power distribution、CSI reconstruction、path consistency、array consistency 和可选 alignment 分量。
- [x] 4.2 为每个 loss 分量实现 weight、enabled flag、valid mask、available count 和 unavailable reason diagnostics。
- [x] 4.3 新增 CSI NMSE、path MAE/Gain NMSE、normalized beamforming gain 和 condition/town/scene grouped report helper。
- [x] 4.4 在 run metadata/final config 中记录 enabled modalities、physics loss weights、array/codebook source、sensitive usage flags 和 `main_conclusion_eligible`。
- [x] 4.5 添加 synthetic loss/backward、missing target skip、metric grouping 和 target oracle eligibility tests。

## 5. 配置与 CLI

- [x] 5.1 新增 `configs/fusion/physics_informed_mmw_debug.yaml`，使用 `data.dataset.type: mmw`、`model.primary.type: pinn_multimodal_beam` 和最小 debug 训练设置。
- [x] 5.2 新增 canonical hybrid 配置和 ablation overlay：no-physics、no-CSI-reconstruction、no-path-loss、no-array-consistency、no-physics-head、CSI-only、image-only、image+CSI、full multimodal。
- [x] 5.3 将包内 inspection CLI 或 debug shape summary 接入 pyproject console script 或现有 CLI module，并添加 help smoke。
- [x] 5.4 确保所有新配置使用现有 `kd-sensing-train` / `kd-sensing-evaluate` 入口，不新增根目录 `train.py`、`evaluate.py` 或 `scripts/inspect_dataset.py`。
- [x] 5.5 运行 `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py -q`，覆盖新增配置加载和 retired-route guard。

## 6. 文档与治理

- [x] 6.1 更新 README 的简短 quickstart/入口说明，明确使用 `kd-sensing-train` 和包内 inspection，不写根脚本命令。
- [x] 6.2 更新 `docs/mainline_model_catalog.md`、`docs/experiment_protocols.md`、`docs/experiment_matrix.md`、`docs/result_claims_registry.md` 和 `docs/model_architecture_inventory.md`，记录 physics-informed baseline、ablation matrix、claim pending 状态和 sensitive usage boundary。
- [x] 6.3 更新 `docs/project_surface_inventory.md` 或 focused architecture tests，登记 `pinn_multimodal_beam` whole-model exception、owner 模块、output boundary 和验证命令。
- [x] 6.4 确认文档不把 CSI/path/beam-power oracle run 写成 MMW sensor-assisted 主结论。

## 7. 验证

- [x] 7.1 运行 `openspec validate add-physics-informed-mmw-beam-baseline --strict`。
- [x] 7.2 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`，验证入口、轻量导入、whole-model exception 和本地产物边界。
- [x] 7.3 运行新增 focused tests：`conda run -n kd_mm_beam pytest tests/test_physics_informed_mmw.py -q`。
- [x] 7.4 运行 CLI smoke：`conda run -n kd_mm_beam kd-sensing-train --help` 和新增 inspection/help 命令。
- [x] 7.5 如真实 MMW 数据可用，运行只读 inspection smoke；如不可用，记录未运行原因并保留 synthetic forward/loss/backward 证据。

## 8. CSI 标签泄漏边界修订

- [x] 8.1 更新 OpenSpec，将 current full CSI 明确拆为 `csi_target`，受限/历史观测拆为 `csi_input`，并记录 `oracle_full` 授权与 warning 规则。
- [x] 8.2 在 MMW dataset/physics adapter 中实现 `csi_input`、`csi_target`、`beam_label`、`beam_power` 和结构化 `path_params` 输出，默认不暴露当前完整 CSI 输入。
- [x] 8.3 在 batch/model 输入准备中拒绝把 `csi_target` 传入 forward，loss 仅从 `csi_target` 读取 CSI reconstruction 监督。
- [x] 8.4 新增 `vision_only`、`partial_csi_multimodal`、`history_csi_multimodal`、`oracle_full_csi` 配置，并校正 full multimodal 默认不启用 CSI 输入。
- [x] 8.5 补充 README 和 focused tests，覆盖 partial/history/oracle 模式、oracle guard、配置加载和无 CSI 输入默认行为。
