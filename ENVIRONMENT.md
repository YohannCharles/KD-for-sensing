# 环境快照

> 这是 2026-07-15 的一次性采样记录，不代表当前机器、未来 runner 或论文复现环境。运行前应重新探测硬件和依赖，不能从本文推断资源可用性。

项目命令的唯一长期约束是使用 `kd_mm_beam`：

```bash
conda run -n kd_mm_beam <command>
```

本次采样结果：

- Linux 6.8.0，Python 3.11.15
- PyTorch 2.12.1+cu130，CUDA runtime 13.0
- 采样时 CUDA 可用，共检测到 8 张 NVIDIA A40

这些值只用于解释同日的本地验证上下文。真实训练仍需核对 driver、显存、数据布局和 checkpoint；所有输出继续写入 ignored 的 `outputs/`、`logs/` 或 `outputs/cache/`。

源码级验证入口是 `make verify-quick`、`make verify-cli-config`、`make verify-compile` 和 `make verify-full`。本文不额外声称 coverage、lint、type check 或 GPU training gate 已启用。
