# BeamBench baseline patch notes

## 本仓库新增内容

- 新增 `src/kd_sensing/baselines/beambench/`：
  - `dataset_check.py`：只读 CSV/path/label/identifier checker
  - `metrics.py`：官方非环形 DBA/top-k 与本仓库 circular metric adapter
  - `mock.py`：显式 `MOCK` dataset 生成器
  - `official.py`：官方仓库审计和 `challenge.py` eval plan wrapper
  - `pipeline.py`：极小 mock train/eval/checkpoint 闭环
  - `image_ae_gps.py`：Arnold22 BeamBench Table III `Camera=AE, GPS=Direct, Fusion=Yes` 本地训练实现，并包含论文 split runner、官方 GPS distance-angle 特征、冻结 AE latent cache、AMP/TF32、fused AdamW 和 DataLoader 并行优化
- 新增包内 CLI：
  - `kd_sensing.cli.beambench_check_dataset`
  - `kd_sensing.cli.beambench_train_baseline`
  - `kd_sensing.cli.beambench_eval_baseline`
  - `kd_sensing.cli.train_beambench_image_ae_gps`
  - `kd_sensing.cli.run_beambench_image_ae_gps_tableiii`
- 新增薄脚本：
  - `scripts/check_dataset.py`
  - `scripts/train_baseline.py`
  - `scripts/eval_baseline.py`
  - `scripts/train_beambench_image_ae_gps.py`
  - `scripts/run_beambench_image_ae_gps_tableiii.py`
- 新增配置：`configs/baselines/beambench_reproduction.yaml`
- 新增论文目标配置：`configs/fusion/beambench_image_ae_gps_direct.yaml`
- 新增测试：
  - `tests/test_beambench_dataset_check.py`
  - `tests/test_beambench_image_ae_gps_direct.py`
  - `tests/test_beambench_image_ae_encoder.py`
  - `tests/test_beambench_metrics.py`
  - `tests/test_beambench_mock_pipeline.py`
- 更新架构边界：
  - `tests/test_architecture_boundaries.py`
  - `docs/project_surface_inventory.md`
- 更新 `.gitignore`，继续忽略 `results/*` 运行产物，但允许跟踪 `results/reproduce_baseline.md` 这一份复现记录。

## 官方代码修改

没有修改、vendoring 或复制官方 BeamBench 源码。官方仓库只在 `/tmp/beambench-official` 临时 clone 后审计。

## 可比性影响

- 官方真实评估：未执行，保持 blocked；因此没有产生可比较官方结果。
- Image AE + GPS Direct 本地训练：使用本仓库 `CameraAutoEncoder` 和本地 DeepSense6G sequence CSV，结构贴合论文目标行，但不使用官方 pretrained AE/fusion 权重和官方完整超参搜索流程；只能作为本地复现路径，不能直接声称等同 Table III 数值。
- 论文 split runner：按用户纠正后的协议在 scenes 32-34 联合训练并评估 scenes 31-34；支持 `validation` 与 `test_as_validation` 两种 best checkpoint 选择口径，后者只作为本地 upper-bound，不等同官方 unseen test。
- GPS Direct 特征：新增并修正 `paper_distance_angle` 对齐官方 `challenge.py` 的 `[distance, calibrated_angle_deg]` 二维输入；角度使用官方 `arctan(x/y)`，scene32 校准角使用 `-0.8125375604986421 + pi/2 = 0.7583`。`paper_calibrated_relative_polar` 保留为三维 ablation。不同 GPS feature mode 会改变模型输入维度和数值可比性，run report 中会记录。
- Feature cache：cache signature 新增 GPS 特征版本和 scene 校准角，避免旧 `atan2` 或旧 scene32 校准角生成的 frozen AE latent/GPS cache 被复用。
- Camera AE：默认本地 paper runner 改为重新训练 512 维 AE latent，以贴近官方 Camera AE encoder 输出维度；scene31 专项实验显示 512d AE 配合 GPS 修复可把 scene31 `official_top3_dba` 提升到 `0.6824`。
- Eval-only：Table III runner 新增 `--fusion-checkpoint`，可直接加载已有 paper-split checkpoint 评估 scenes 31-34 并生成 Table III CSV/Markdown/JSON 汇总，不重新训练 fusion。
- 训练加速：冻结 AE latent cache 不改变 frozen encoder 的数学输入输出，只避免 fusion 阶段重复读图和重复 encoder forward；AMP/TF32 是 CUDA 吞吐优化，默认启用但可通过 `--no-amp --no-tf32` 关闭以做完全 fp32 调试。
- mock smoke：使用本仓库 `TinyBeamBenchClassifier` 和生成的 mock CSV，只验证 data loading、forward、loss、metric、checkpoint save/load 和 evaluation，不与官方结果比较。
- metric：新增字段明确区分 `official_*` 非环形口径与 `circular_*` 64-beam 环形口径，避免混用。

## 回滚方式

如需回滚本 change，可删除：

- `src/kd_sensing/baselines/beambench/`
- `src/kd_sensing/cli/beambench_*`
- `scripts/check_dataset.py`
- `scripts/train_baseline.py`
- `scripts/eval_baseline.py`
- `scripts/train_beambench_image_ae_gps.py`
- `scripts/run_beambench_image_ae_gps_tableiii.py`
- `configs/baselines/beambench_reproduction.yaml`
- `configs/fusion/beambench_image_ae_gps_direct.yaml`
- `tests/test_beambench_*`
- 本 change 新增的根目录复现文档和 `results/reproduce_baseline.md`

并恢复 `.gitignore`、`tests/test_architecture_boundaries.py` 和 `docs/project_surface_inventory.md` 中对应条目。
