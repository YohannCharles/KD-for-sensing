## Context

runtime cleanup 和 sample cache overwrite 都会对配置或 manifest 提供的路径调用递归删除，但现有校验没有完整覆盖空路径、项目根、scan root、符号链接和 Git 查询失败。MMW archive preparation 在覆盖目录前只使用弱摘要并直接 `extractall`；run index 会把 `/proc` command line 原样写入公共 artifact；checkpoint 和 TinyViT 下载的完整性语义依赖 PyTorch 版本默认值。

这些入口跨 diagnostics、preprocessing、data 和 model owner，但共同原则是：任何删除、解压、反序列化、下载或公开进程参数的操作都必须在副作用前建立可验证信任边界，并在边界不可证明时 fail closed。

## Goals / Non-Goals

**Goals:**

- 递归删除只能作用于明确允许、可证明属于当前 workflow 的路径。
- archive extraction 在覆盖目标前完成路径、资源上限与完整摘要校验。
- 公共 diagnostics artifact 不泄露命令行凭证或 secret override。
- checkpoint/远程权重默认采用安全反序列化与完整性验证。
- 预处理同名碰撞、全量失败和非原子写不再静默成功。

**Non-Goals:**

- 不删除、迁移或重建用户现有 dataset、cache、outputs、logs 或 checkpoint。
- 不新增通用安全框架、外部依赖或后台服务。
- 不保证第三方 checkpoint 可在安全模式下无迁移直接加载；例外必须显式 trusted-local。

## Decisions

### 1. 每个破坏性 owner 在副作用前执行 containment 校验

cleanup 使用 manifest 的允许根和 project root；sample cache 使用配置解析后的 cache root 和 owner marker。两者都校验 lexical path 与 `Path.resolve()` 后的 containment，拒绝空路径、根自身、项目根、dataset root 和符号链接逃逸。相比一个全局 path-safety framework，owner 内的窄 helper 更容易审计且避免错误共享不同策略。

### 2. Cache ownership 由小型 JSON marker 证明

新建 cache 时写入包含 schema、owner、resolved root 和创建时间的 marker。overwrite 只接受 marker 与当前 owner/root 匹配，且 `data.mdb`、`lock.mdb` 都是普通非符号链接文件的 LMDB 目录。旧无 marker cache 或只有伪造 marker 的空目录保留不动，用户可换新目录重建；不自动认领历史目录。

### 3. Cleanup manifest 与 Git 状态都 fail closed

scan manifest 为每个候选记录 `regular_file`、`directory`、`symlink` 或 `other` 文件系统类型；apply 在解析路径和删除前通过 `lstat` 复核类型，并继续校验 manifest schema/rules version、非空 path、scan root containment、mtime/size/protection 状态和 tracked files。`git ls-files` 非零退出、项目根不可解析、类型漂移或 manifest 缺字段都 fail closed，而不是把 tracked 集合视为空。

### 4. ZIP 先完整预检，再安全逐项解压

在任何 `rmtree` 或写入之前遍历 central directory，拒绝绝对路径、`..`、resolved target 越界、过多 member、超限解压大小和异常压缩比。marker 使用完整 archive SHA256 与算法版本；解压逐项写入临时目录，成功后原子替换目标。

### 5. Command line 保留 argv，公开前统一脱敏

读取 `/proc/*/cmdline` 时按 NUL 分隔保留 argv list；复用现有命令脱敏规则处理 token、password、credential、URI userinfo 和敏感 override。公共 JSON 只保存脱敏 argv/展示字符串，不保存 raw bytes。

### 6. Checkpoint 默认安全加载，unsafe 是显式例外

所有 state-dict/tensor checkpoint 显式使用 `weights_only=True`。确需 legacy pickle 时，调用方必须传入 `trusted_local=true` 并在 metadata/warning 中记录；远程或来源不明 artifact 不允许 unsafe。TinyViT 网络下载默认关闭，只有固定 HTTPS URL 与明确 SHA256 同时存在时才可下载并在加载前校验。

### 7. Preprocessor 以稳定资源 identity 和失败汇总收口

输出键基于受控输入根下的规范化相对路径，而不是 basename。批量处理聚合有限失败样本，零成功或超过明确阈值时返回失败；CSV/metadata 使用已有原子写 helper。内部配置验证不捕获宽泛 `Exception` 后继续。

### 8. RGB image-derived cache 保留为可选等价加速路径

当前 dataset、cache policy、preprocess CLI 和七份 MMW/DeepSense 配置都消费版本化 RGB/ImageNet image-derived cache；它不同于已删除的 image motion mask cache。保留该 current performance surface，但要求 cache key 覆盖相对资源 identity、image size、profile 和 transform version，命中结果必须与直接 RGB/ImageNet transform 的 shape、dtype 和数值语义一致。无 image modality 时不得访问；旧 `image_motion_*` 字段继续 fail closed。

## Risks / Trade-offs

- [旧 manifest/cache 被拒绝] -> 保留原文件，只要求重新扫描或重建到新目录，不自动删除。
- [安全 checkpoint 模式拒绝自定义对象] -> 仅为明确 trusted-local legacy artifact 提供 opt-in，并记录风险。
- [ZIP 预检增加准备时间] -> 完整 SHA256 和 central-directory 扫描只在 archive digest 变化时执行。
- [预处理由部分成功改为失败可能影响旧脚本] -> 保留有限错误报告和可配置非零阈值，但零成功始终失败。

## Migration Plan

1. 先补临时目录安全测试，再修改 cleanup/cache 删除路径。
2. 增加新 marker/schema；旧 artifact 只读保留。
3. 切换 MMW extraction、run index 和 checkpoint loader，补兼容测试。
4. 收紧预处理错误与原子写，运行 focused/quick/full regression。

回滚只能恢复代码行为，不得自动删除新旧 artifact；任何 unsafe checkpoint 或无 marker 删除行为的恢复都需新的 OpenSpec change。

## Open Questions

- 无。TinyViT 未提供受验证 SHA256 时默认要求本地 checkpoint。
