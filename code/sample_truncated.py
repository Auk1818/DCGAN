"""
截断采样（truncation trick）—— 不重训，直接用现有 generator.ckpt 出图

原理：训练时 z ~ N(0, 1)。采样时把 z 的幅度缩小（z = psi * z，psi < 1），
只从隐空间"密度高"的中心区域采样，避开训练时很少见过的边缘区域。
代价是多样性略降，收益是坏样本明显减少。来自 Brock 等人 2019 年的 BigGAN。

用法：
    # 对比 psi = 1.0 / 0.8 / 0.7 / 0.5，固定噪声与训练脚本一致（seed=1 的前 24 个）
    python sample_truncated.py --ckpt ~/runs/final_d/generator.ckpt --out ~/runs/final_d/trunc

    # 换一批随机噪声看更多样本
    python sample_truncated.py --ckpt ... --out ... --seed 7 --n 32 --cols 8
"""

import os
import argparse
import numpy as np
import mindspore as ms
from mindspore import nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_args():
    p = argparse.ArgumentParser("DCGAN 截断采样")
    p.add_argument("--ckpt", required=True, help="generator.ckpt 或 generator_ema.ckpt")
    p.add_argument("--out", required=True, help="输出目录")
    p.add_argument("--psi", type=float, nargs="+", default=[1.0, 0.8, 0.7, 0.5],
                   help="截断系数，可传多个，每个出一张图")
    p.add_argument("--n", type=int, default=24, help="每张图的样本数")
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--seed", type=int, default=1, help="=1 时与训练脚本的 fixed_noise 完全一致")
    p.add_argument("--nz", type=int, default=100)
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--norm", default="pm1", choices=["01", "pm1"])
    p.add_argument("--device", default="CPU", choices=["GPU", "CPU"])
    return p.parse_args()


def build_generator(nz, ngf):
    """与 train_dcgan_v2 / v3 完全相同的结构，保证 ckpt 能对上"""
    from mindspore.common.initializer import Normal
    wi = Normal(mean=0, sigma=0.02)
    gi = Normal(mean=1, sigma=0.02)
    nc = 3

    class Generator(nn.Cell):
        def __init__(self):
            super().__init__()
            self.generator = nn.SequentialCell(
                nn.Conv2dTranspose(nz, ngf * 8, 4, 1, "valid", weight_init=wi),
                nn.BatchNorm2d(ngf * 8, gamma_init=gi), nn.ReLU(),
                nn.Conv2dTranspose(ngf * 8, ngf * 4, 4, 2, "pad", 1, weight_init=wi),
                nn.BatchNorm2d(ngf * 4, gamma_init=gi), nn.ReLU(),
                nn.Conv2dTranspose(ngf * 4, ngf * 2, 4, 2, "pad", 1, weight_init=wi),
                nn.BatchNorm2d(ngf * 2, gamma_init=gi), nn.ReLU(),
                nn.Conv2dTranspose(ngf * 2, ngf, 4, 2, "pad", 1, weight_init=wi),
                nn.BatchNorm2d(ngf, gamma_init=gi), nn.ReLU(),
                nn.Conv2dTranspose(ngf, nc, 4, 2, "pad", 1, weight_init=wi),
                nn.Tanh())

        def construct(self, x):
            return self.generator(x)

    return Generator()


def to_display(img, norm):
    if norm == "pm1":
        img = (img + 1.0) / 2.0
    return np.clip(img, 0, 1)


def save_grid(imgs, path, cols, norm, title=None):
    rows = int(np.ceil(len(imgs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5), dpi=100)
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for i, im in enumerate(imgs):
        axes[i // cols, i % cols].imshow(to_display(im, norm))
    if title:
        fig.suptitle(title, y=0.995)
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = get_args()
    os.makedirs(args.out, exist_ok=True)
    try:
        ms.set_context(mode=ms.PYNATIVE_MODE)
        ms.set_device(args.device)
    except AttributeError:
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device)

    G = build_generator(args.nz, args.ngf)
    params = ms.load_checkpoint(os.path.expanduser(args.ckpt))
    missing, _ = ms.load_param_into_net(G, params)
    if missing:
        print("[警告] 以下参数没有从 ckpt 载入:", missing)
    G.set_train(False)

    # 与 train_dcgan_v2 的 fixed_noise 同源：RandomState(seed).randn(24, nz, 1, 1)
    z_base = np.random.RandomState(args.seed).randn(args.n, args.nz, 1, 1).astype("float32")

    strip = []   # 每个 psi 取前 cols 张，最后拼一张总览
    for psi in args.psi:
        z = ms.Tensor(z_base * psi)
        imgs = G(z).transpose(0, 2, 3, 1).asnumpy()
        save_grid(imgs, os.path.join(args.out, f"psi_{psi:.2f}.png"), args.cols, args.norm,
                  title=f"truncation psi = {psi}")
        strip.append(imgs[:args.cols])
        print(f"psi={psi:.2f}  ->  psi_{psi:.2f}.png")

    # 总览：每行一个 psi，同一列是同一个 z，方便看截断对同一张脸的影响
    fig, axes = plt.subplots(len(args.psi), args.cols,
                             figsize=(args.cols * 1.5, len(args.psi) * 1.55), dpi=100)
    axes = np.atleast_2d(axes)
    for r, (psi, row) in enumerate(zip(args.psi, strip)):
        for c in range(args.cols):
            axes[r, c].axis("off")
            axes[r, c].imshow(to_display(row[c], args.norm))
        axes[r, 0].set_title(f"psi={psi}", fontsize=8, loc="left")
    plt.savefig(os.path.join(args.out, "truncation_overview.png"), bbox_inches="tight")
    plt.close(fig)
    print("总览图: truncation_overview.png")


if __name__ == "__main__":
    main()
