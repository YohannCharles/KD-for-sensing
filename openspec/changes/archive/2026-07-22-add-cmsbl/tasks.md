## 1. P0 主线与历史边界

- [x] 1.1 更新 CMSBL delta specs、current research brief、surface inventory、maintainer context 和 retired-route 结论。
- [x] 1.2 删除已证伪/停止路线的 current specs、旧 canonical YAML、analysis、scripts 和对应 tests，保持 `outputs/` 完全不变。
- [x] 1.3 收缩架构脚本白名单并验证 public CLI 仍只有 train/evaluate/preprocess。

## 2. P1 U-Mask 与 BCACL U2 收缩

- [x] 2.1 删除 U-Mask 的 PCER、PGCD、candidate dynamic Router、cached reroute 和只服务历史诊断的 intermediate payload。
- [x] 2.2 将 BCACL model/config/loss 收缩为 aux-joint U2 private/shared supervision，删除 prototype/teacher/quality/two-stage。
- [x] 2.3 收缩 MMW/DeepSense6G T2 的默认关闭 BCACL/CMSBL 配置并保持 clean clone parse。

## 3. P2 最小 CMSBL

- [x] 3.1 实现单一 linear auxiliary schedule 和 disabled-path 等价。
- [x] 3.2 实现 standalone Top-1 capacity JSON 校验、train-only EMA、有界 modality weights 和 checkpoint state。
- [x] 3.3 实现 canonical mask ID、per-sample fusion/BPA raw loss、15-mask EMA/count、有界权重和 checkpoint state。
- [x] 3.4 每 epoch 写单一 JSON 并复用 TensorBoard 标量；validation/test 不更新状态。

## 4. P3 测试与归档

- [x] 4.1 增加最小 CMSBL focused tests，覆盖 linear schedule、capacity provenance/EMA、mask mapping/weights、resume 和 disabled path。
- [x] 4.2 归档完成、失败或未启动的前序 changes，不将 retired specs 合并回 current specs。
- [x] 4.3 运行 `openspec validate --all --strict`、architecture/config/CLI/compile、CMSBL/BCACL focused tests 和 `make verify-quick`。
- [x] 4.4 运行 `conda run -n kd_mm_beam pytest -q`，确认源码 diff 不包含 `outputs/`、cache、dataset、日志或 checkpoint。
