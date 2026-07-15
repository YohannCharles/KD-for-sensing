## 1. 递归删除边界

- [x] 1.1 让 runtime cleanup apply 拒绝空路径、项目根、scan root、允许根外路径、符号链接逃逸和 manifest 版本/状态漂移。
- [x] 1.2 让 cleanup 的 Git/project-root 查询失败时 fail closed，并使用 `conda run -n kd_mm_beam` 增加破坏性边界回归测试。
- [x] 1.3 为 sample cache 新建 owner marker，overwrite 仅允许删除受控 cache root 下匹配 marker 与结构的目录。
- [x] 1.4 使用 `conda run -n kd_mm_beam` 增加 cache 项目根、dataset root、无 marker、伪 marker和 symlink escape 测试。

## 2. Archive 与进程信息

- [x] 2.1 为 MMW ZIP 增加 member path、数量、大小、压缩比和完整 SHA256 预检，并使用临时目录原子发布。
- [x] 2.2 使用 `conda run -n kd_mm_beam` 增加 traversal、资源上限、stale marker、失败保留旧目标测试。
- [x] 2.3 让 run index 保留 argv 边界并统一脱敏公共 resources/summary/card，不写 raw cmdline。
- [x] 2.4 使用 `conda run -n kd_mm_beam` 增加含空格路径、token/password、URI userinfo 和普通参数测试。

## 3. Checkpoint 与预处理

- [x] 3.1 审计并将 state-dict/tensor checkpoint 显式切换到安全加载；仅为明确 trusted-local legacy artifact 保留 opt-in。
- [x] 3.2 让 TinyViT 网络权重默认关闭，只有固定 HTTPS URL 与 SHA256 同时存在才下载和发布缓存。
- [x] 3.3 修复 radar/image/MMW 批量预处理的 basename 碰撞、宽泛异常吞没、零成功和非原子写，并移除 config validation 的 fail-open 捕获。
- [x] 3.4 使用 `conda run -n kd_mm_beam` 运行 checkpoint、TinyViT、preprocess、cache 与 config validation focused tests。

## 4. 回归

- [x] 4.1 运行 `openspec validate harden-destructive-runtime-operations --strict`、`openspec validate --all --strict`、`make verify-quick`、`make verify-cli-config` 和 `make verify-compile`。
- [x] 4.2 运行 `conda run -n kd_mm_beam pytest -q` 并修复本 change 引入的回归。
- [x] 4.3 复核 `git status --short`，确认没有删除或纳入真实 dataset、outputs、logs、cache、checkpoint、权重和临时安全测试产物。
