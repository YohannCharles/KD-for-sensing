## Why

当前 MMW Town10 `l5p6` 准备产物使用滑窗后随机窗口切分，导致 train/test 中大量相邻窗口共享历史帧、未来帧和未来 beam 标签序列。该协议会让验证曲线接近训练曲线，也会让不同场景同模态曲线看起来高度重合，无法作为严格泛化或跨场景结论依据。

## What Changes

- 将 MMW Town10 sequence split 从默认随机窗口切分收紧为可审计的 group-safe split，按连续片段、agent、时间块或等价 group 单位分配 train/test，确保同一 group 不跨 split。
- 为 MMW split metadata 增加协议字段和泄漏诊断，包括 split strategy、group key、guard band、最小帧间隔、train/test window overlap、相邻窗口比例和未来标签序列复用统计。
- 不再提供旧随机窗口切分的准备/公开 split builder 兼容路径；已有或未知协议产物通过 split metadata eligibility 保守处理，避免进入主结论或 strict validation。
- 收紧 MMW scenario-LOSO 中 target_adapt/target_test 的时间邻近防护，避免 few-shot target_adapt 与 target_test 共享或高度重叠的滑窗上下文。
- 更新训练和评估运行产物，使结果能显示 group-safe split 协议、strict eligibility 和泄漏诊断，并支持下游 summary 过滤不可比结果。
- 不改变 MMW 模型、loss、指标数学定义或公开训练/评估 CLI 参数；必要时新增 split strategy 配置和公开准备入口参数。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `mmw-town10-dataset-preparation`: 将 MMW sequence split 契约从“窗口随机切分”收紧为 group-safe split，并要求输出泄漏诊断 metadata。
- `mmw-cross-scene-adaptation-protocol`: target_adapt/target_test 防泄漏从 sample id 无交集扩展到时间邻近和滑窗上下文隔离。
- `experiment-workflow`: 训练、评估和 summary 产物必须记录 split 协议与泄漏诊断，并标记 unknown 或 strict-ineligible split 的主结论资格。

## Impact

- 主要影响 `src/kd_sensing/data/mmw/preparation.py`、`scripts/mmw/build_sequence_splits_from_manifest.py`、MMW 数据准备配置、`scripts/run_mmw_sunny_modal15_l5p6_h246.sh` 和相关测试。
- 可能影响 `src/kd_sensing/engine/run_metadata.py`、训练日志/runtime metadata、standalone evaluate 报告和 MMW quick summary 过滤逻辑。
- 需要为旧 `l5p6` 产物和新 group-safe 产物提供清晰的 split tag 或 metadata 区分，避免 TensorBoard 与 metrics 横向比较时混用协议。
- 不提交或迁移 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 等本地产物；验证只使用小型 fixture 或临时目录。
