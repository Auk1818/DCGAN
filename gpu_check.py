#!/usr/bin/env python3
"""
MindSpore GPU 可用性验证脚本

用途：在 WSL2 (Linux) 里验证 MindSpore 能否真正使用你的 NVIDIA 显卡。
      RTX 40 系（sm_89）不在 MindSpore 预编译的架构列表里，只能靠 PTX 即时编译兜底，
      所以必须实测，不能假设能用。

用法（务必在 WSL 里运行，不是 Windows CMD）：
    wsl                                  # 从 Windows 进入 WSL
    conda activate ms
    python gpu_check.py

首次运行会卡住几分钟（驱动在即时编译几百个算子），属于正常现象，不要中断。
编译结果会缓存在 ~/.nv/ComputeCache，第二次运行就很快了。
"""

import os
import sys
import time
import platform

BAR = "=" * 62


def fail(msg, hint=""):
    print("\n[失败] " + msg)
    if hint:
        print("       " + hint)
    print(BAR)
    sys.exit(1)


print(BAR)
print("MindSpore GPU 可用性验证")
print(BAR)

# ---------- 0. 环境自检：必须在 Linux (WSL) 里 ----------
print("\n[0/6] 检查运行环境")
print("      操作系统 :", platform.system(), platform.release())
print("      Python   :", sys.version.split()[0])

if platform.system() == "Windows":
    fail(
        "你正在 Windows 上运行这个脚本。",
        "MindSpore 的 Windows 包不含 CUDA 后端，在这里永远看不到显卡。\n"
        "       请先在终端敲 wsl 回车进入 Linux，再运行本脚本。",
    )

# WSL 的标志：/proc/version 里含 microsoft
in_wsl = False
try:
    with open("/proc/version") as f:
        in_wsl = "microsoft" in f.read().lower()
except Exception:
    pass
print("      是否 WSL :", "是" if in_wsl else "否（原生 Linux 也可以）")

# ---------- 1. 驱动直通 ----------
print("\n[1/6] 检查显卡驱动直通")
if in_wsl and not os.path.exists("/usr/lib/wsl/lib/libcuda.so.1"):
    print("      [警告] 没找到 /usr/lib/wsl/lib/libcuda.so.1")
    print("             说明 WSL 的显卡直通可能没生效。先在 Windows 里跑 wsl --update")
rc = os.system("nvidia-smi --query-gpu=name,memory.total,driver_version "
               "--format=csv,noheader 2>/dev/null")
if rc != 0:
    fail(
        "nvidia-smi 跑不起来，WSL 看不到显卡。",
        "1) 在 Windows PowerShell 里执行 wsl --update 然后 wsl --shutdown\n"
        "       2) 确认 Windows 侧装了较新的 NVIDIA 驱动\n"
        "       3) 千万不要在 WSL 内部安装 NVIDIA 驱动，那会把直通弄坏",
    )

# ---------- 2. CUDA 运行库 ----------
print("\n[2/6] 检查 CUDA 运行库")
for lib, why in [
    ("libcudart.so.11.0", "CUDA 11 运行时"),
    ("libcublas.so.11", "矩阵运算库"),
    ("libcudnn.so.8", "深度学习算子库"),
]:
    found = os.popen(f"ldconfig -p 2>/dev/null | grep -c {lib}").read().strip()
    conda_hit = ""
    prefix = os.environ.get("CONDA_PREFIX", "")
    if prefix and os.path.exists(prefix + "/lib"):
        n = os.popen(f"ls {prefix}/lib 2>/dev/null | grep -c '{lib.split('.so')[0]}'").read().strip()
        if n and n != "0":
            conda_hit = "（在 conda 环境里找到）"
    status = "✓" if (found != "0" or conda_hit) else "✗"
    print(f"      {status} {lib:22s} {why} {conda_hit}")

if os.environ.get("CONDA_PREFIX") and os.environ.get("CONDA_PREFIX") + "/lib" not in os.environ.get("LD_LIBRARY_PATH", ""):
    print("\n      [提示] LD_LIBRARY_PATH 里没有 conda 的 lib 目录。")
    print("             如果下面报找不到库，先执行：")
    print("             export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH")

# ---------- 3. 导入 MindSpore ----------
print("\n[3/6] 导入 MindSpore 并切换到 GPU")
try:
    import numpy as np
    import mindspore as ms
    from mindspore import nn, ops, Tensor
except Exception as e:
    fail("导入 MindSpore 失败：%s" % e, "先执行 pip install mindspore==2.6.0")

print("      MindSpore 版本 :", ms.__version__)
try:
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="GPU")
except Exception as e:
    fail("切换到 GPU 后端失败：%s" % e,
         "说明这个 MindSpore 包里没有 GPU 插件，或者是 Windows 版")
print("      后端切换成功")

print("\n      ↓ 下面是真正的考验。首次运行会卡几分钟（驱动在即时编译），请耐心等待")

NO_KERNEL_HINT = (
    "出现 'no kernel image' 说明你的显卡架构(sm_89)没有兜底代码，这条路走不通。\n"
    "       不要再折腾了，直接用免费云 GPU：Google Colab 或 魔搭 ModelScope。"
)

# ---------- 4. 基础算子 ----------
print("\n[4/6] 测试基础算子")
try:
    t0 = time.time()
    a = Tensor(np.random.randn(512, 512).astype("float32"))
    r = ops.matmul(a, a)
    _ = r.asnumpy()
    print("      ✓ 矩阵乘        输出 %s   (%.1f 秒)" % (r.shape, time.time() - t0))
except Exception as e:
    fail("矩阵乘失败：%s" % e, NO_KERNEL_HINT)

# ---------- 5. DCGAN 实际用到的算子 ----------
print("\n[5/6] 测试 DCGAN 实际用到的算子")
try:
    x = Tensor(np.random.randn(8, 3, 64, 64).astype("float32"))

    t0 = time.time()
    conv = nn.Conv2d(3, 64, 4, 2, "pad", 1)
    y = conv(x); _ = y.asnumpy()
    print("      ✓ 卷积(判别器)   输出 %s   (%.1f 秒)" % (y.shape, time.time() - t0))

    t0 = time.time()
    convT = nn.Conv2dTranspose(64, 32, 4, 2, "pad", 1)
    z = convT(y); _ = z.asnumpy()
    print("      ✓ 转置卷积(生成器) 输出 %s   (%.1f 秒)" % (z.shape, time.time() - t0))

    t0 = time.time()
    bn = nn.BatchNorm2d(32)
    bn.set_train()
    w = bn(z); _ = w.asnumpy()
    print("      ✓ BatchNorm      输出 %s   (%.1f 秒)" % (w.shape, time.time() - t0))

    t0 = time.time()
    loss = nn.BCELoss(reduction="mean")
    p = ops.sigmoid(Tensor(np.random.randn(8, 1).astype("float32")))
    t = Tensor(np.ones((8, 1), dtype="float32"))
    l = loss(p, t); _ = l.asnumpy()
    print("      ✓ BCE 损失       值 %.4f   (%.1f 秒)" % (float(l.asnumpy()), time.time() - t0))
except Exception as e:
    fail("DCGAN 算子测试失败：%s" % e, NO_KERNEL_HINT)

# ---------- 6. 反向传播 + 速度实测 ----------
print("\n[6/6] 测试反向传播并实测速度")
try:
    class MiniG(nn.Cell):
        def __init__(self):
            super().__init__()
            self.net = nn.SequentialCell(
                nn.Conv2dTranspose(100, 256, 4, 1, "valid"),
                nn.BatchNorm2d(256), nn.ReLU(),
                nn.Conv2dTranspose(256, 128, 4, 2, "pad", 1),
                nn.BatchNorm2d(128), nn.ReLU(),
                nn.Conv2dTranspose(128, 3, 4, 2, "pad", 1),
                nn.Tanh())
        def construct(self, z):
            return self.net(z)

    g = MiniG()
    opt = nn.Adam(g.trainable_params(), learning_rate=2e-4, beta1=0.5)

    def forward(z):
        out = g(z)
        return ops.reduce_mean(out * out)

    grad_fn = ms.value_and_grad(forward, None, opt.parameters)

    def step(z):
        v, grads = grad_fn(z)
        opt(grads)
        return v

    z = Tensor(np.random.randn(128, 100, 1, 1).astype("float32"))
    t0 = time.time()
    v = step(z); _ = v.asnumpy()
    print("      ✓ 首次反向传播完成 (%.1f 秒，含即时编译)" % (time.time() - t0))

    N = 10
    t0 = time.time()
    for _ in range(N):
        v = step(z)
    _ = v.asnumpy()
    per = (time.time() - t0) / N
    print("      ✓ 稳定后每步耗时 %.3f 秒 (batch=128)" % per)

    est = per * 549 * 40 / 60.0
    print("\n      按这个速度估算：549 batch/轮 × 40 轮 ≈ %.0f 分钟" % est)
    if est < 90:
        print("      → 很快，可以放心跑多组消融实验")
    elif est < 300:
        print("      → 可接受，消融实验建议每组只跑 10 轮")
    else:
        print("      → 偏慢，考虑减小 batch 或用免费云 GPU")
except Exception as e:
    fail("反向传播测试失败：%s" % e, NO_KERNEL_HINT)

print("\n" + BAR)
print("全部通过：你的显卡可以跑这个 DCGAN 项目")
print(BAR)
print("\n下一步：把 train_dcgan.py 里的")
print('    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")')
print("改成")
print('    ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU")')
print("\n数据路径也要改成 WSL 能访问的形式，例如 /mnt/d/_SEU_/CV_proj/DCGAN/faces")
