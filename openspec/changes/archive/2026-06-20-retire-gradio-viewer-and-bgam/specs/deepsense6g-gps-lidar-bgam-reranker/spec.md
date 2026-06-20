## REMOVED Requirements

### Requirement: DeepSense6G GPS+LiDAR BGAM reranker workflow
**Reason**: DeepSense6G GPS+LiDAR BGAM 已确认不再使用，不再作为 current reranker workflow、paper reproduction 或诊断路线维护。
**Migration**: 无兼容迁移；使用仍保留的 supervised/adaptation、Image+GPS JEPA、MMW GPS v2、CSI hardening 或其它 current workflow。

### Requirement: GPS/RSU geometry prior
**Reason**: 该 requirement 在本 capability 中只服务 BGAM manifest、mask/gate 和 reranker 输入。
**Migration**: 若未来其它 current workflow 需要 GPS/RSU AoD 几何，必须在对应 capability 中重新定义通用几何契约。

### Requirement: BGAM manifest enrich
**Reason**: BGAM manifest enrich 只服务已退役 BGAM workflow。
**Migration**: 无兼容迁移；不得保留 BGAM manifest enrich CLI、配置或 helper 作为当前入口。

### Requirement: GPSLidarBGAMDataset
**Reason**: BGAM dataset 只服务已退役 BGAM workflow。
**Migration**: 删除专属 dataset 和 focused tests；保留 workflow 不应依赖 BGAM batch 字段。

### Requirement: LiDAR spatial encoder
**Reason**: 本 requirement 下的 LiDAR spatial encoder 是 BGAM 专属模型组件。
**Migration**: 保留的 LiDAR 模态或其它模型如需 encoder，应使用各自 current spec 或新 change 定义。

### Requirement: GPSGuidedBGAM mask/gate
**Reason**: BGAM mask/gate 是退役路线核心组件。
**Migration**: 无兼容迁移；不得以 registry alias、model helper 或 debug mask 形式恢复。

### Requirement: beam angle table
**Reason**: 该 beam angle table 契约在本 capability 中只服务 BGAM candidate-to-angle/mask 解释。
**Migration**: 通用 beam label 或 circular metric 语义由其它保留 specs 继续承担。

### Requirement: LiDAR BEV cross-attention
**Reason**: 该 cross-attention 模块是 BGAM reranker 专属实现。
**Migration**: 不迁移；未来 LiDAR cross-attention 模型必须通过新的 model capability 提出。

### Requirement: GPS prior encoder
**Reason**: 该 GPS prior encoder 是 BGAM reranker 专属组件。
**Migration**: 保留的 GPS v2/adapter workflow 不应导入该 BGAM encoder。

### Requirement: GPSLidarBGAMBeamPredictor
**Reason**: BGAM beam predictor 属于退役 workflow。
**Migration**: 删除模型、loss wiring 和 tests；不提供兼容模型类型。

### Requirement: BGAM training losses
**Reason**: BGAM loss 只服务退役 BGAM workflow。
**Migration**: 删除专属 loss 和 focused tests；保留通用 Top-K/circular metrics 不受仅因字符串命中而删除。

### Requirement: BGAM training and evaluation protocol
**Reason**: BGAM 训练、评估和 ablation matrix 已退役。
**Migration**: 无兼容迁移；历史输出只作为本地产物或 archive 背景。

### Requirement: BGAM evaluation artifacts
**Reason**: BGAM metrics、predictions、debug mask 和 comparison report 随 workflow 退役。
**Migration**: 不再生成或验证 BGAM 产物；本地既有产物不在本 change 中删除。

### Requirement: BGAM anti-leakage guard
**Reason**: 防泄漏 guard 只约束已退役 BGAM 数据/模型路径。
**Migration**: 保留 workflow 继续使用各自的数据泄漏和 split guard；不迁移 BGAM 专属断言。

### Requirement: BGAM validation and documentation
**Reason**: BGAM 单元测试、CPU smoke、CLI help 和 README 工作流说明随能力退役删除。
**Migration**: 使用保留 workflow 的 focused validation 和最终 `conda run -n kd_mm_beam pytest -q`。
