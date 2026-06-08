# BeamBench 复现环境记录

## 官方要求

官方仓库：`https://github.com/ITU-AI-ML-in-5G-Challenge/BeamBench`

临时 clone：`/tmp/beambench-official`

审计 commit：`8e2c29a2afc898a69b9f9f7ece039d1e48ba60e8`

官方 README 推荐评估命令：

```bash
python3 challenge.py --gpu_id 0 --data_folder ./raw_data/test/ --csv ml_challenge_test_multi_modal.csv
```

官方 Dockerfile 要求：

- Ubuntu 18.04
- CUDA 11.4，base image 为 `nvidia/cuda:11.4.2-runtime-ubuntu18.04`
- Python 3.7
- `torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113`
- 关键依赖：`numpy`、`Pillow`、`matplotlib`、`utm`、`opencv-python`、`sklearn`、`tqdm`、`pandas`、`future`、`open3d`、`h5py`

官方默认路径：

- 数据目录：`./raw_data/test/`
- CSV：`ml_challenge_test_multi_modal.csv`
- 模型目录：`results/models`
- 预测输出：`results/topk`

## 当前 kd_mm_beam 环境

采集命令：

```bash
conda run -n kd_mm_beam python -c "import sys, torch; import torchvision; print('python=' + sys.version.replace(chr(10), ' ')); print('torch=' + torch.__version__); print('torchvision=' + torchvision.__version__); print('cuda_runtime=' + str(torch.version.cuda)); print('cuda_available=' + str(torch.cuda.is_available())); print('cuda_device_count=' + str(torch.cuda.device_count())); print('gpu=' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'))"
```

结果：

- Python：`3.11.15 | packaged by conda-forge | (main, Mar 5 2026, 16:45:40) [GCC 14.3.0]`
- PyTorch：`2.11.0+cu130`
- torchvision：`0.26.0+cu130`
- CUDA runtime：`13.0`
- CUDA available：`True`
- GPU count：`4`
- GPU 0：`NVIDIA GeForce RTX 3090`

## 版本偏差与最小可运行方案

当前环境能运行本仓库 mock checker、mock baseline、metric 单测和 wrapper help，但不是官方原始环境。主要偏差是 Python 3.11 vs 3.7、CUDA 13.0 vs 11.4、PyTorch 2.11/cu130 vs 官方 cu113 wheel。

真实官方复现当前 blocked：

- 本地没有官方 `raw_data/test/ml_challenge_test_multi_modal.csv`
- 本地没有官方 `results/models/*.pth` 权重
- 官方仓库中多个 `challenge.py` 引用的模型/配置只有 `.pyc` 或缺失 `.py`
- 若要严格复现官方结果，建议使用官方 Dockerfile 或等价 Ubuntu 18.04/CUDA 11.4/Python 3.7 环境，并补齐官方数据和权重

本 change 的最小可运行方案是：在 `kd_mm_beam` 中运行 read-only dataset checker、BeamBench metric helper、官方 eval plan wrapper 和显式 `MOCK` pipeline smoke。
