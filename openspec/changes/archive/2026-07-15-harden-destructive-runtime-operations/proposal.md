## Why

当前两个显式删除入口在恶意或误配置输入下可能递归删除项目根或任意路径，且 cleanup 在 Git 状态不可用时会 fail open；MMW archive、checkpoint 下载/加载和诊断进程信息也缺少一致的信任边界。必须先封住数据丢失与敏感信息暴露风险，再继续扩大自动化工作流。

## What Changes

- **BREAKING**：runtime cleanup apply 拒绝空路径、项目根、scan root、允许根外路径和 schema/version 不匹配；无法确认 Git tracked 状态时拒绝删除。
- **BREAKING**：sample cache overwrite 仅允许删除受控 cache root 下、具有匹配 marker 的 cache 目录，拒绝项目根、dataset root、符号链接逃逸和任意用户目录。
- MMW ZIP 解压在删除/覆盖前验证所有 member 的 resolved path、数量、总大小和压缩比，使用完整 archive digest，禁止 path traversal 与 stale marker 复用。
- run index 对 `/proc/*/cmdline` 保留 argv 边界并统一脱敏，公共 JSON 不输出原始凭证或 secret override。
- checkpoint 默认按 tensor/state-dict 安全模式加载；需要 pickle 的 trusted-local 例外必须显式声明。TinyViT 网络权重使用固定来源和 SHA256 校验，未校验下载默认关闭。
- 预处理器统一使用稳定资源 identity、原子写和失败阈值；同名输入碰撞、全量失败或内部验证异常不得静默成功。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `runtime-artifact-cleanup`：删除执行增加根路径、manifest 与 Git fail-closed 防护。
- `automated-cache-policy`：cache overwrite 增加允许根、marker 和路径 containment 契约。
- `image-preprocessing-profiles`：明确 RGB/ImageNet image-derived cache 是可选等价加速路径，并继续拒绝已退役 image motion cache。
- `mmw-town10-dataset-preparation`：archive extraction 增加 traversal、资源上限和完整 digest 校验。
- `experiment-run-index`：进程命令行必须保留 argv 语义并在公共 artifact 中脱敏。
- `tinyvit-image-encoder`：预训练权重下载和 checkpoint 加载需要固定来源与完整性校验。
- `project-health-guardrails`：checkpoint 与预处理信任边界必须 fail closed，不允许静默部分成功。

## Impact

- 影响 runtime cleanup、sample cache、MMW preparation、run index、artifact/checkpoint loader、TinyViT 和 image/radar/MMW preprocessing owner。
- 旧 cleanup manifest、无 marker cache、弱摘要 extraction marker 和未校验 TinyViT 缓存将被拒绝并要求重新生成。
- 不删除任何现有 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 或历史权重；测试只使用临时目录和 synthetic artifact。
