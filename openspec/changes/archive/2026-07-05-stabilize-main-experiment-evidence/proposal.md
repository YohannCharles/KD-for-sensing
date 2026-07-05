## Why

当前主线实验已经从“继续扩展新模块”转向“收敛证据”：Scene31-34 缺失模态主实验需要完成 classifier/external-lite/final summary，JEPA predictive/shortcut 需要真实 checkpoint manifest 才能从 smoke 升级为可讨论证据。该 change 将主实验证据收敛作为独立路线，避免继续扩大研究表面。

## What Changes

- 明确 Scene31-34 主实验的 final evidence checklist：core proto n=5、classifier baseline、external-lite maskfix、final all summary、missing-count 曲线、compute profile 和 paper tables。
- 明确 JEPA/Predictive robustness 从 smoke 到 real benchmark 的升级条件：audited checkpoints、strict comparable clean anchor、stress curves、difficulty digest、Image ResNet+GPS margin。
- 将“冻结主方法候选、补证据而非继续加模块”写入对应 specs 和 tasks。
- 不新增模型结构，不恢复退役 KD/HiST/Top8/residual/BGAM/viewer 路线。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `scenes31-34-main-missing-modality-workflow`: 增加 final evidence checklist 和停止扩表边界。
- `predictive-jepa-robustness`: 增加 real benchmark promotion gate。
- `jepa-gps-shortcut-benchmark`: 增加真实 checkpoint manifest 的 strict comparability 要求。
- `mainline-experiment-documentation`: 增加主实验证据收敛记录要求。

## Impact

- 主要影响 Scene31-34 scripts/docs、JEPA benchmark manifest/docs、claim registry 和 focused validation。
- 真实训练和 benchmark 输出仍写 ignored `outputs/`。
- 不提交 metrics、figures、checkpoint 或 cache。
