## 1. OpenSpec 与冲突清理

- [ ] 1.1 更新 `openspec/changes/add-resnet18-image-architecture` 中所有 specs、design 和 tasks，删除 legacy motion、`motion_mask`、image motion cache 与 `motion_cnn` 兼容要求。
- [ ] 1.2 用 `rg -n "image_motion|motion_mask|motion_cnn|legacy_motion_cnn|load_motion_masks|build_motion_mask_pair"` 建立当前可执行引用清单，排除 archive 历史记录和本变更说明。
- [ ] 1.3 确认删除范围不包含 `outputs/`，并避免任务或脚本清理历史训练产物。

## 2. 配置、Cache 与预处理入口

- [ ] 2.1 删除 `configs/preprocess/image_motion_cache.yaml` 和 README/docs/run.sh 中的 image motion cache 预热、清理、推荐命令。
- [ ] 2.2 从 `src/kd_sensing/config/defaults.py`、配置解析和 validation 中删除 `image_motion_*` 默认字段，并让旧字段或 `image_profile: motion_mask` 触发清晰错误。
- [ ] 2.3 从 `src/kd_sensing/engine/cache_policy.py` 删除 image cache policy 解析，仅保留 LiDAR BEV 等仍支持 cache 的模态。
- [ ] 2.4 从 `src/kd_sensing/engine/run_metadata.py`、final config 和报告写出逻辑中删除 `image_motion_*` metadata。
- [ ] 2.5 删除 `src/kd_sensing/preprocessing/image.py` 的 image motion preprocessor，并从 `src/kd_sensing/preprocessing/__init__.py`、`src/kd_sensing/cli/preprocess.py` 移除对应导出和 CLI choice。

## 3. Dataset、Batch 与模型架构

- [ ] 3.1 从 `src/kd_sensing/data/transform_ops/image.py`、`cache.py` 和 `_legacy.py` 删除 image motion cache key、metadata、motion mask 构造、懒加载和兼容导出。
- [ ] 3.2 更新 `src/kd_sensing/data/datasets/scenario9.py`，让 image modality 只走 RGB/ImageNet loader，并删除构造参数、属性和 `_resolve_image_motion_cache_dir`。
- [ ] 3.3 更新 batch 准备逻辑，删除 `motion_mask` 专用 shape/padding 分支，保持 image batch 与 RGB/ImageNet encoder 契约一致。
- [ ] 3.4 从 `src/kd_sensing/models/image_encoders.py`、`models/__init__.py` 和 `models/modular.py` 删除 `MotionCNNImageEncoder`、`motion_cnn`/`legacy_motion_cnn` 注册名和默认分支。
- [ ] 3.5 更新 image-only、image+radar fusion、CRAF、MARF 和 checkpoint registry/default config，确保默认路径不引用旧 motion branch 或旧 image checkpoint。

## 4. 诊断、文档与示例

- [ ] 4.1 更新 Gradio viewer manifest/prediction 相关模块，image processed 表示改为 RGB/ImageNet 输入或其可视化派生物，不再读取 image motion cache。
- [ ] 4.2 更新 README、`docs/training_throughput.md` 和可视化说明，删除 image motion cache、motion mask 和 legacy image branch 说明。
- [ ] 4.3 更新 `.gitignore` 或产物边界文档中关于 cache 的描述，保留 `outputs/` 不删除的约束。

## 5. 测试重写

- [ ] 5.1 删除或重写 image motion cache、motion profile、motion encoder 和 legacy checkpoint 兼容测试。
- [ ] 5.2 新增 RGB image dataset/model smoke tests，覆盖 image-only 和包含 image 的 fusion 配置不访问 image motion cache。
- [ ] 5.3 新增配置解析回归测试，覆盖 `image_motion_*`、`motion_mask`、`motion_cnn` 和 `legacy_motion_cnn` 被拒绝。
- [ ] 5.4 使用 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py tests/test_resnet18_image_architecture.py tests/test_student_configs.py` 验证受影响测试。

## 6. 最终验证

- [ ] 6.1 使用 `rg -n "image_motion|motion_mask|motion_cnn|legacy_motion_cnn|load_motion_masks|build_motion_mask_pair" src configs tests docs README.md run.sh openspec/changes/add-resnet18-image-architecture` 确认当前源码、配置、测试、文档和 active change 不再残留可执行引用。
- [ ] 6.2 使用 `openspec validate remove-image-motion --strict` 验证本变更 artifacts。
- [ ] 6.3 使用 `conda run -n kd_mm_beam pytest` 运行全量测试；若运行时间过长，记录已跑子集和剩余风险。
