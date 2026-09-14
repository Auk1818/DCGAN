# 在 WSL2 里让 MindSpore 用上 RTX 40 系显卡

> 目标：不花钱、不用华为云，用本机的 RTX 40 系显卡跑 DCGAN 训练。
> 预计耗时：验证 20 分钟，全部装完 40~60 分钟。

---

## 先理解为什么必须这么绕

三件事叠在一起，才导致不能直接在 Windows 上用显卡：

1. **MindSpore 的 Windows 安装包里没有 CUDA 后端。**
   拆开 `mindspore-2.10.0-cp310-cp310-win_amd64.whl`（1557 个文件）逐个看过，
   只有 `mindspore_cpu.dll`，没有任何 gpu / cuda 相关的动态库。
   代码里 `device_target="CPU"` 不是作者的选择，是 Windows 下唯一能填的值。

2. **Linux 安装包里有 GPU 后端，但只到 CUDA 11.6。**
   `mindspore/lib/plugin/libmindspore_gpu.so.11.1` 和 `.11.6`，最新的 2.10 版本依然如此，没有 CUDA 12。

3. **CUDA 11.6 原生不支持 40 系（sm_89），但有 PTX 兜底。**
   拆 `libcuda_ops.so.11`（819 MB）数过里面的算子：

   | 类型 | 覆盖架构 |
   |---|---|
   | 预编译机器码 cubin（2030 个） | sm_60 / 61 / 70 / 75 / 80 —— **最高到 A100，没有 sm_89** |
   | 中间代码 PTX（407 个） | compute_60 / **compute_86** |

   PTX 是中间代码，显卡驱动能在运行时把它现场编译成 sm_89 的机器码。
   **所以 40 系有兜底路径，但必须实测才知道通不通** —— MindSpore 自己的算子有 PTX，
   但它还依赖 NVIDIA 的 cuBLAS / cuDNN，那两个有没有兜底无法离线确认。

---

## 三条铁律（踩了就翻车）

| 铁律 | 原因 |
|---|---|
| **所有操作都在 WSL 里做** | Windows 上的 conda 环境再怎么配也看不到显卡 |
| **绝对不要在 WSL 内部安装 NVIDIA 驱动** | 驱动由 Windows 侧提供，WSL 通过直通层访问。在 WSL 里装驱动会把直通彻底弄坏，这是最常见的翻车点 |
| **用 conda 装 CUDA，不要装系统级 CUDA** | CUDA 11.6 官方 deb 源只支持到 Ubuntu 20.04。你的 WSL 大概率是 22.04 / 24.04，装不上。conda 装在虚拟环境里，不挑发行版 |

---

## 第 0 步：进入 WSL

在 Windows 终端（CMD 或 PowerShell）里：

```
wsl
```

回车后提示符应该从 `C:\Users\38703>` 变成类似 `你的用户名@机器名:~$`。
**看到 `$` 结尾的提示符才算进去了。** 后面所有命令都在这里面敲。

顺便确认一下是 WSL2 而不是 WSL1（在 Windows 终端里查）：

```
wsl -l -v
```

`VERSION` 那一列必须是 `2`。如果是 1，执行 `wsl --set-version <发行版名> 2` 升级。
**WSL1 不支持显卡直通**，必须是 2。

---

## 第 1 步：确认显卡直通（这步不过，后面全白搭）

在 WSL 里：

```bash
nvidia-smi
```

**能看到你的 RTX 40 系型号和显存 = 直通已通。** Win11 自带，不需要额外配置。

如果报 command not found 或看不到显卡，回 Windows 终端执行：

```
wsl --update
wsl --shutdown
```

然后重新 `wsl` 进去再试。还是不行就是 Windows 侧的 NVIDIA 驱动太老，去官网更新。

---

## 第 2 步：装 Miniconda（如果 WSL 里还没有）

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
exec bash
```

注意：**WSL 里的 conda 和 Windows 里的 conda 是两套独立的东西**，
Windows 里已经有 conda 不代表 WSL 里有。

---

## 第 3 步：建环境 + 装 CUDA 11.6

```bash
conda create -n ms python=3.9 -y
conda activate ms
conda install -c conda-forge cudatoolkit=11.6 cudnn=8.4.1 -y
```

这一步会下载约 2~3 GB，装在 conda 环境目录里，不碰系统。

装完让 MindSpore 能找到这些库：

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

**这一行每开一个新终端都要重敲。** 想一劳永逸就写进配置：

```bash
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
```

---

## 第 4 步：装 MindSpore

```bash
pip install mindspore==2.6.0
```

选 2.6.0 而不是最新的 2.10 是因为它更成熟稳定，GPU 插件的坑更少。
装完快速确认一下：

```bash
python -c "import mindspore; print(mindspore.__version__)"
```

---

## 第 5 步：跑验证脚本（关键的一步）

WSL 里访问 D 盘的路径是 `/mnt/d/...`，所以：

```bash
cd /mnt/d/_SEU_/CV_proj/DCGAN
python gpu_check.py
```

脚本会依次测：环境 → 驱动直通 → CUDA 库 → 基础算子 → DCGAN 专用算子 → 反向传播与速度。

**首次运行会卡几分钟不动**，那是显卡驱动在即时编译几百个算子，
属于正常现象，**不要 Ctrl+C 中断**。编译结果缓存在 `~/.nv/ComputeCache`，第二次就快了。

### 怎么判断结果

**全部打勾** → 成功。脚本最后会告诉你每步耗时和 40 轮的预估时间。

**出现 `no kernel image is available for execution on the device`** → 失败。
说明 cuBLAS / cuDNN 那层没有 sm_89 的兜底代码，这条路走不通。
**别再折腾了**，直接跳到最后一节的免费云 GPU 方案。

**出现找不到 libcudart / libcudnn** → 是 `LD_LIBRARY_PATH` 没设好，回第 3 步重设再试。

---

## 第 6 步：改训练脚本

验证通过后，`train_dcgan.py` 里要改两处：

```python
# 1. 后端：CPU → GPU
ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU")

# 2. 路径：Windows 盘符 → WSL 挂载路径
data_path   = '/mnt/d/_SEU_/CV_proj/DCGAN/faces'
output_dir  = '/mnt/d/_SEU_/CV_proj/DCGAN/result'
```

**性能提示**：`/mnt/d/` 是跨文件系统访问，读 7 万张小图会明显变慢。
如果发现数据加载是瓶颈（GPU 利用率上不去），把数据集拷到 WSL 内部再训：

```bash
cp -r /mnt/d/_SEU_/CV_proj/DCGAN/faces ~/faces
```

结果还是写回 `/mnt/d/...`，方便在 Windows 里直接看。

---

## 如果 40 系这条路走不通：免费云 GPU

两个都不要钱：

| 平台 | 显卡 | 说明 |
|---|---|---|
| **Google Colab** | T4（sm_75） | 在 MindSpore 的预编译架构列表里，兼容性完美，不用赌 PTX 能不能兜底。需要梯子 |
| **魔搭 ModelScope** | 有免费 GPU 时长 | 国内直连，不用梯子 |

Colab 上的安装命令（Colab 默认带 CUDA，可能需要降级到 11.x）：

```bash
!pip install mindspore==2.6.0
```

然后把 `faces.zip` 上传或直接在 Notebook 里下载：

```python
from download import download
download("https://download.mindspore.cn/dataset/Faces/faces.zip", "./faces", kind="zip")
```

---

## 值不值得花这个时间

| 方案 | 40 轮预估耗时 |
|---|---|
| Windows CPU（你们上次的方式） | 十几个小时，而且跑到第 41 轮崩了 |
| RTX 40 系 + WSL2 | 大概率半小时到一小时 |

这个差距决定报告能做到什么程度：
CPU 上你只能跑出一组基线，GPU 上能跑四五组消融对比。
**报告里有没有对比实验，分量完全不一样。**
