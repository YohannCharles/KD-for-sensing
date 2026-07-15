## 1. Temporal split 完整性

- [x] 1.1 将 H5/P1 launcher 切换为现有 group-safe sequence split，并拒绝重叠窗口的逐样本拆分。
- [x] 1.2 为 split artifact 增加 sequence、sample、历史输入帧和 target 帧两两不相交审计及有限冲突诊断。
- [x] 1.3 使用 `conda run -n kd_mm_beam` 增加并运行 H5/P1 launcher、split identity 和 provenance focused tests。

## 2. Validation 与 final test 隔离

- [x] 2.1 删除 trainer 的 test-as-validation fallback，并在选模启用但 validation 缺失时 fail closed。
- [x] 2.2 支持显式 fixed-epoch/no-selection 无 validation 运行：跳过逐轮 validation/best checkpoint，final checkpoint 使用 `last.pth`。
- [x] 2.3 使用 `conda run -n kd_mm_beam` 增加并运行 early stopping、checkpoint 与 test loader 不被训练迭代的 focused tests。

## 3. Train-only normalization

- [x] 3.1 统一遍历 Dataset、Subset、ConcatDataset 的 leaf/effective train indices，并让所有已有模态统计只从实际训练子集拟合。
- [x] 3.2 修复 pooled dataset、normalization artifact 和 position metric 的首 leaf/顶层读取问题，增加 feature mode 与 fingerprint 校验。
- [x] 3.3 使用 `conda run -n kd_mm_beam` 增加并运行 GPS、LiDAR、mmWave、CSI、position 及 pooled normalization focused tests。

## 4. Validation loss 与配置迁移

- [x] 4.1 将 evaluation pass 的 validation loss 改为按有效 sample/token 数加权，并覆盖不等长 batch。
- [x] 4.2 更新受影响 canonical/current 配置：提供独立 validation，或显式 fixed-epoch/no-selection；不得继续隐式使用 test。
- [x] 4.3 使用 `conda run -n kd_mm_beam` 运行 evaluation、config characterization 和 CLI/config focused tests。

## 5. Evidence 与回归

- [x] 5.1 将修复前 H5/P1 evidence 标为 `not_comparable`，补齐 claim/protocol/history 的泄漏 caveat 与重跑 gate，不写入新数值。
- [x] 5.2 修复 H5/P1 provenance schema 测试漂移，并运行相关 summary/evaluator tests。
- [x] 5.3 运行 `openspec validate enforce-evaluation-data-integrity --strict`、`openspec validate --all --strict`、`make verify-quick`、`make verify-cli-config`、`make verify-compile` 和 `conda run -n kd_mm_beam pytest -q`。
- [x] 5.4 复核 `git status --short`，确认未纳入 dataset、outputs、logs、cache、checkpoint、PDF 或其它本地产物。
