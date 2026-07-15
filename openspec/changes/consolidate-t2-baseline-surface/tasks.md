## 1. 固化 T2/baseline 输入

- [x] 1.1 从已验证的 T2/S1/RMBP resolved input 提炼无 runtime metadata 的 tracked MMW shared base 与三个小 overlay。
- [x] 1.2 修改 all-weather matrix、T2 hyperparameter screen 与 BPA/CMA helpers，使其只读取 tracked recipe，不再读取 `outputs/` YAML。
- [x] 1.3 使用 `conda run -n kd_mm_beam pytest` 增加并运行 canonical recipe/launcher focused tests，覆盖无 `outputs/` 的 config dry-run。

## 2. 退役外部蒸馏与兼容面

- [ ] 2.1 删除 external `teacher_guidance` runtime、配置解析、optimizer/trainer hook、测试和文档引用。
- [ ] 2.2 从 U-Mask/config/runtime 删除未启用的 full-to-partial、weak-pattern KD、teacher checkpoint/tensor 与 legacy compatibility fields，同时保留 embedded teacher CE 和 temporal superset consistency。
- [ ] 2.3 使用 `conda run -n kd_mm_beam pytest` 运行 T2/S1/U-Mask focused tests，覆盖启用与禁用 same-model consistency。

## 3. 收缩共享闭包

- [ ] 3.1 将 MMW runtime 所需的 loader/transform 依赖从 DeepSense6G、CSI、physics 和已退役 modal surface 中显式分离。
- [ ] 3.2 收缩 registry、factory、validation、CLI 和 package exports，使 T2/baseline import graph 不再无条件加载 retired family。
- [ ] 3.3 使用 `conda run -n kd_mm_beam pytest` 运行 MMW/T2/baseline synthetic forward、config load 与 import boundary tests。

## 4. 删除非 T2/baseline surface

- [ ] 4.1 删除非 T2/baseline public console scripts、MMW GPS v2/physics、Scene31-34/final-C2 diagnostics 及其专属 configs、tests、docs references。
- [ ] 4.2 删除不再可达的 C2、DeepSense6G workflow、CSI、physics、历史 baseline、pretraining、diagnostic source/config/script/test families；每批删除前反向检查 callers。
- [ ] 4.3 更新 `pyproject.toml`、CLI lifecycle 和 architecture guard，使仅 train/evaluate/preprocess 为 public CLI，T2 scripts 均有 lifecycle。
- [ ] 4.4 使用 `conda run -n kd_mm_beam pytest` 运行 architecture、retired-route、CLI help、T2/baseline focused tests，并修复 stale imports/references。

## 5. 文档与 current specs

- [ ] 5.1 更新 README、inventory、导航、maintainer index、研究/claim/protocol/model 文档，使 T2/baseline 成为唯一 current surface。
- [ ] 5.2 写入集中历史说明，记录退役 family 的原用途和 OpenSpec archive/git history 追溯方式；不保留兼容 stub。
- [ ] 5.3 将本 change 的 delta 同步到 current specs，并删除已退役 capability 的 current lifecycle 描述。

## 6. 验证与收尾

- [ ] 6.1 使用 `conda run -n kd_mm_beam` 运行 T2/baseline config/forward/launcher focused tests、`make verify-quick`、`make verify-cli-config` 和 `make verify-compile`。
- [ ] 6.2 运行 `openspec validate consolidate-t2-baseline-surface --strict`、`openspec validate --all --strict`，并使用 `conda run -n kd_mm_beam pytest -q` 执行最终回归。
- [ ] 6.3 检查 `git status --short`，确认不引入 dataset、outputs、logs、cache、checkpoint 或其它 runtime artifact，并记录未运行验证与剩余风险。
