## Why

当前项目已经具备 clean CSI、pilot estimation noise 和已完成的 CSI channel degradation 能力，但 `CSI模态处理对比实验.md` 的目标不是继续做破坏性退化，而是寻找“最终上限接近 clean teacher、但收敛更慢、在多模态 joint training 中更容易被 easy modality 压制”的 CSI 设置。现在需要把这一实验叙事固化为可复现的 CSI hardening 控制变量矩阵、配置入口和统一分析脚本。

## What Changes

- 新增 information-preserving CSI hardening：common phase、subcarrier phase slope、antenna calibration 和 fixed antenna permutation，默认关闭，作用于 RMS 归一化后的 complex CSI，避免改变 clean future beam label。
- 扩展 `pilot_dual_view_csi` encoder，使其支持 `use_internal_gru: false`、view gate warmup、delay view warmup、tokenizer 容量消融和 hardening diagnostics。
- 增加 CSI-only 控制变量配置矩阵：A 组 clean/pilot/destructive negative control，B 组只改 hardening，C 组只改 encoder，D 组 hardening+encoder 组合。
- 增加 easy modality + CSI 验证配置：GPS-only、GPS+clean CSI、GPS+slow CSI、CSI-prioritized warmup、GPS+slow CSI 的 G2D-style 配置。
- 新增 sweep 分析脚本，统一计算 final last10、best、E50/E80/E90、ceiling gap、E90 ratio、destructive 判定和 slow-high-ceiling 候选排序。
- 将已有 `csi_degradation` 作为 destructive negative control 或鲁棒性对照使用，不替代新的 information-preserving hardening 主线。

## Capabilities

### New Capabilities

- `csi-hardening-experiment-matrix`: 定义 CSI hard-to-learn 控制变量矩阵、推荐运行顺序、候选筛选指标、分析脚本输出和多模态验证配置要求。

### Modified Capabilities

- `csi-modality-model`: 扩展 CSI encoder 的信息保留型 hardening、内部 GRU 开关、双视图 warmup、tokenizer 消融和 diagnostics 行为。
- `g2d-multimodal-distillation`: 明确 G2D teacher ensemble、confidence ranking 和 SMP 能处理包含 `csi` 的模态集合，用于 GPS+CSI 等 easy+CSI 验证。

## Impact

- 受影响代码：
  - `src/kd_sensing/models/csi.py`
  - `src/kd_sensing/models/modular.py`
  - `src/kd_sensing/engine/trainer.py`
  - `src/kd_sensing/engine/g2d_training.py`
  - `src/kd_sensing/distillation/g2d.py`
  - `src/kd_sensing/distillation/g2d_smp.py`
  - `src/kd_sensing/config/io.py`
  - `src/kd_sensing/config/canonical.py`
  - `src/kd_sensing/engine/run_metadata.py`
  - `configs/csi/*.yaml`
  - `configs/fusion/*.yaml`
  - `scripts/analyze_csi_hardening_sweep.py`
- 受影响测试：
  - `tests/test_csi_modality.py`
  - `tests/test_student_configs.py`
  - `tests/test_g2d_smp.py`
  - `tests/test_g2d_distiller.py`
  - 新增 CSI hardening sweep 分析脚本测试
- 不新增外部依赖；分析脚本使用项目现有 Python 依赖。
- 默认行为保持兼容：未配置 `csi_hardening`、warmup 或 `use_internal_gru: false` 时，现有 clean CSI 和 degraded CSI 配置输出不变。
