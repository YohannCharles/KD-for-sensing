## Why

PGCD 的动态权重在连续退化下几乎不优于 train-fit global mean，但现有证据无法区分 prototype bank/feature collapse、质量信息擦除、模态惰性和 Router 输入不可观测。需要对已有 validation-best checkpoint 做固定 inner-only 离线诊断，在不重训和不读取 channel/path 数据的前提下给出可复核的层级证据。

## What Changes

- 增加 A0、A1、B2、C0、C7 的统一 checkpoint 与 layer manifest，并显式记录不存在的公平 no-prototype 因果对照。
- 增加只使用 inner-train/inner-validation、按 sample identity 确定的 clean/corrupt 配对样本与分片特征缓存。
- 增加 prototype geometry、层级 collapse/probe、cross-modal alignment、modality dependence、Router observability 和 beam-conditional corruption collapse 诊断。
- 为 `UMaskBeamJEPA` 增加默认关闭的只读 intermediate return，复用现有 forward 与 PGCD corruption generator，不引入训练分支。
- 增加可恢复的 GPU 抽取、CPU 统计和报告聚合脚本；所有输出保持 single-seed、inner/development、claim-ineligible，且不自动训练任何模型。

## Capabilities

### New Capabilities

- `prototype-collapse-diagnostics`: 定义固定 checkpoint、样本、缓存、分层指标、BC1-BC7 判定与本地诊断产物契约。

### Modified Capabilities

- `u-mask-beam-jepa`: 增加默认关闭、只读且不改变训练/推理语义的中间特征导出契约。

## Impact

变更涉及 `src/kd_sensing/models/u_mask_beam_jepa.py` 的 opt-in debug return、`analysis/` 下的离线诊断程序、`scripts/` 下的本地 runner 和聚焦测试。实现复用已安装的 PyTorch、NumPy、pandas、scikit-learn 与 matplotlib，不新增依赖、不修改历史 checkpoint、不读取 outer test 做 probe/统计，也不把 `outputs/`、日志或缓存纳入源码。
