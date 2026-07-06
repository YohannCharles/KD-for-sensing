# Tasks

- [x] 1. 新增 `lightweight_patchvit_frame` / `patchvit_frame` encoder registry wrapper，复用已有 `patch_vit` token encoder。
- [x] 2. 新增 Scene31-34 PatchViT ablation generator，产出 2 个单模态预训练配置、2 个 downstream 配置和 manifest。
- [x] 3. 添加 focused tests 覆盖 encoder forward 与 generator 配置语义。
- [x] 4. 运行 focused validation 与 OpenSpec validate。
- [x] 5. 在不关闭现有 TinyViT 训练的前提下，启动 PatchViT image/lidar 并行预训练，并在 checkpoint 就绪后并行启动 downstream。
