## Why

当前 C0-C4 x D0-D7 uniform Mean DBA 更像通用 Image+GPS 鲁棒性 sanity：大量 clean/mild cell 对强监督 CNN+GPS 有利，导致 JEPA 下游即使在极端块有局部优势，也难以形成相对 CNN+GPS 的 5 个百分点主 claim。需要新增一个真正考察 JEPA 预测表征价值的主场景：当前图像局部/整帧不可观测、GPS 可能可信但错误、历史上下文可用于预测当前 beam-relevant latent。

## What Changes

- 新增 `predictive-jepa-robustness` 能力，定义 Predictive Robustness benchmark 的目标、场景、模型组、主指标和 claim 边界。
- 新增 JEPA predictive hybrid fusion 模型线：结合 mean/content query、GPS residual query、temporal predicted latent 和 feature-consistency gate；该 gate 不读取 C/D condition id，不作为 CxD router。
- 扩展 JEPA downstream pooler/adapter 能力，支持 hybrid pooler、content+GPS residual query、temporal predicted latent auxiliary branch 和内部一致性 diagnostics。
- 扩展 difficulty pipeline，新增预测鲁棒性 profile：current-frame missing/history-available、beam-relevant semantic occlusion、novel weather/domain shift、plausible wrong GPS 等输入扰动；所有扰动仍不得移动 target、beam power、sample id 或 split metadata。
- 扩展 JEPA GPS shortcut benchmark runner，使其支持 predictive robustness suite、regional aggregate、CNN+GPS 对照 margin、overall CxD sanity 和 claim 状态输出。
- 新增派生配置与 smoke/real-run 账本条目；真实训练、评估、CSV、PNG、checkpoint 和 cache 仍写入 ignored `outputs/` 或 `logs/`，不进入源码。
- 不删除、不重命名、不替换现有 GPS-only、Image CNN+GPS、Image AE+GPS、JEPA GPS-biased、JEPA GPS-query-pool、Scenario D CxD workflow。
- 不新增旧式 root script、不复制训练循环、不恢复退役 KD/HiST/residual 路线。

## Capabilities

### New Capabilities
- `predictive-jepa-robustness`: 定义 JEPA 预测表征主场景、predictive robustness suite、JEPA predictive hybrid fusion 模型线、相对 CNN+GPS 的主 claim 口径和输出产物边界。

### Modified Capabilities
- `jepa-downstream-extensibility`: 扩展 JEPA downstream pooler/adapter 契约，支持 hybrid pooler、content+GPS residual query、temporal predicted latent auxiliary branch 和 feature-consistency gate diagnostics。
- `modality-difficulty-pipeline`: 扩展 shared difficulty profile/operator 契约，支持 predictive robustness 专用 image/GPS 扰动并保持 no-label-shift、determinism 和 replay metadata。
- `jepa-gps-shortcut-benchmark`: 扩展 benchmark manifest、模型组、aggregation 和产物 schema，支持 predictive robustness suite、regional metric、margin-vs-CNN 和 overall CxD sanity 并列报告。

## Impact

- 受影响代码：
  - `src/kd_sensing/models/jepa_downstream.py`：新增 hybrid/content+GPS residual pooler 或等价可注册 downstream pooler。
  - `src/kd_sensing/models/jepa.py`：暴露 temporal predicted/current latent auxiliary branch 与 consistency diagnostics，保持默认 mean/GPS-query 行为兼容。
  - `src/kd_sensing/models/modular.py` 或可组合 representation core/helper：接收 predictive JEPA auxiliary features 与 feature-consistency gate 输出，普通 core 不被静默改变。
  - `src/kd_sensing/data/difficulty/`：新增 predictive robustness profiles/operators 或在现有 GPS/image operators 上标准化新 condition。
  - `src/kd_sensing/diagnostics/jepa_gps_shortcut_benchmark.py`：新增 predictive robustness suite 解析、aggregation、CSV/JSON/figure manifest 字段。
- 受影响配置：
  - 新增 `configs/fusion/experiments/jepa_image_gps/` 下的 JEPA predictive hybrid fusion 派生配置。
  - 新增或派生 `configs/diagnostics/` 下的 predictive robustness smoke 与 real-run manifest。
- 受影响文档：
  - 更新 `docs/experiment_matrix.md`、`docs/mainline_model_catalog.md`、`docs/experiment_protocols.md`、`docs/result_claims_registry.md`，区分 pending/smoke/real claim。
- 受影响测试：
  - 新增 JEPA pooler/fusion focused tests、difficulty operator tests、benchmark schema/aggregation tests 和配置加载 tests。
  - 继续使用 `conda run -n kd_mm_beam ...` 运行所有项目 Python 测试。
- 不新增外部依赖；不改变 checkpoint 加载默认语义；真实运行产物不得提交。
