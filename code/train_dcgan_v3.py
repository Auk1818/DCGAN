"""
基于 MindSpore 的 DCGAN 生成漫画头像 —— 改进版训练脚本 v3

本版本（v3）在 v2 基础上新增（同样都是开关，默认关闭）：

  [新增] 谱归一化 D         --sn，判别器每个卷积层除以其最大奇异值（Miyato 等 2018），
                            让 D 变平滑，从机制上抑制"D 后期压倒 G"。按 SN-GAN 惯例同时去掉 D 里的 BatchNorm。
  [新增] 生成器 EMA         --ema 0.999，另外维护一份 G 的滑动平均权重用来出图/存盘（Karras 等 2018 ProGAN），
                            样本更平滑，抖动伪影更少。存为 generator_ema.ckpt；generated_epoch*.png 用 EMA 权重出图，
                            同时另存 generated_raw_epoch*.png 便于对比。

相对手册原版 train_dcgan.py 的改动（每一项都可通过命令行开关独立启用/关闭，便于做消融对比）：

  [修复] 随机种子           原版没有种子，任何对比实验都不可复现
  [修复] fixed_noise 固定    原版写在训练循环内，每轮重新采样，导致逐轮对比不可比
  [修复] 数据归一化         --norm pm1 让真实图落在 [-1,1]，与生成器 Tanh 输出对齐
  [修复] 内存堆积           原版 image_list 累积约 250MB 不释放，疑似崩溃原因，已删除
  [修复] 中途落盘           每轮保存 metrics，崩溃不再丢失全部训练数据
  [新增] D(x) / D(G(z)) 记录  比 loss 曲线更能说明博弈是否平衡
  [新增] 单边标签平滑       --label-smooth 0.9，缓解判别器过强
  [新增] TTUR              --d-lr 单独设置判别器学习率
  [新增] 水平翻转增强       --flip，头像左右翻转语义合理，等于数据量翻倍
  [新增] BCEWithLogits     --logits，数值更稳定
  [新增] 断点续训           --resume
  [新增] 隐空间插值图       训练结束自动生成，证明学到的是连续流形而非记忆样本

用法示例：
    # 基线（尽量贴近原版，只加种子）
    python train_dcgan_v3.py --data ~/dcgan_data --out runs/baseline --tag baseline --graph-mode --epochs 40

    # 最终模型 final_d：归一化对齐 + TTUR
    python train_dcgan_v3.py --data ~/dcgan_data --out runs/final_d --tag final_d \
        --graph-mode --norm pm1 --d-lr 1e-4 --epochs 60

    # h_ema：final_d + 生成器 EMA
    python train_dcgan_v3.py --data ~/dcgan_data --out runs/h_ema --tag ema \
        --graph-mode --norm pm1 --d-lr 1e-4 --ema 0.999 --epochs 60

    # f_sn_ema：再加判别器谱归一化（实验中为负结果，见 实验总结.md）
    python train_dcgan_v3.py --data ~/dcgan_data --out runs/f_sn_ema --tag sn_ema \
        --graph-mode --norm pm1 --d-lr 1e-4 --sn --ema 0.999 --epochs 60
"""

import os
import csv
import json
import time
import argparse

import numpy as np
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore import nn, ops
from mindspore.common.initializer import Normal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================ 参数 ============================
def get_args():
    p = argparse.ArgumentParser("DCGAN 漫画头像生成（改进版）")

    p.add_argument("--data", required=True,
                   help="数据集根目录（其下应有一个子目录装着所有 jpg，例如 ~/dcgan_data 下的 faces/）")
    p.add_argument("--out", default="./result", help="输出目录")
    p.add_argument("--tag", default="run", help="本次实验的名字，用于区分不同消融组")
    p.add_argument("--device", default="GPU", choices=["GPU", "CPU"])
    p.add_argument("--graph-mode", action="store_true", help="用静态图模式（更快，但报错难读）")

    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--nz", type=int, default=100, help="隐向量维度")
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--ndf", type=int, default=64)
    p.add_argument("--workers", type=int, default=8, help="数据读取并行数")

    p.add_argument("--lr", type=float, default=2e-4, help="生成器学习率")
    p.add_argument("--d-lr", type=float, default=None,
                   help="判别器学习率，不填则与 --lr 相同。设小一点即 TTUR")
    p.add_argument("--beta1", type=float, default=0.5)

    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--norm", default="01", choices=["01", "pm1"],
                   help="01=原版 x/255 落在[0,1]（与 Tanh 不匹配）；pm1=(x-127.5)/127.5 落在[-1,1]")
    p.add_argument("--flip", action="store_true", help="启用随机水平翻转增强")
    p.add_argument("--label-smooth", type=float, default=1.0,
                   help="判别器真实样本的目标值，1.0=不平滑，常用 0.9")
    p.add_argument("--logits", action="store_true",
                   help="判别器去掉 Sigmoid，改用 BCEWithLogitsLoss（数值更稳）")

    p.add_argument("--sn", action="store_true",
                   help="判别器卷积层加谱归一化，并去掉 D 的 BatchNorm（SN-GAN 惯例）")
    p.add_argument("--sn-keep-bn", action="store_true",
                   help="与 --sn 联用时保留 D 的 BatchNorm（不推荐，仅供消融）")
    p.add_argument("--ema", type=float, default=0.0,
                   help="生成器权重 EMA 的衰减系数，0=关闭，常用 0.999")

    p.add_argument("--resume", action="store_true", help="从上次的 checkpoint 继续训练")
    p.add_argument("--sample-every", type=int, default=1, help="每几轮存一次生成图")
    return p.parse_args()


# ============================ 数据 ============================
def build_dataset(args):
    """返回 (dataset, 每轮 batch 数)"""
    data_path = os.path.expanduser(args.data)
    if not os.path.isdir(data_path):
        raise SystemExit(f"数据目录不存在: {data_path}")

    dataset = ds.ImageFolderDataset(data_path,
                                    num_parallel_workers=args.workers,
                                    shuffle=True,
                                    decode=True)
    tfm = [vision.Resize(args.image_size), vision.CenterCrop(args.image_size)]
    if args.flip:
        tfm.append(vision.RandomHorizontalFlip(0.5))
    tfm.append(vision.HWC2CHW())
    if args.norm == "pm1":
        tfm.append(lambda x: ((x - 127.5) / 127.5).astype("float32"))
    else:
        tfm.append(lambda x: (x / 255).astype("float32"))

    dataset = dataset.project("image").map(tfm, "image").batch(args.batch_size,
                                                               drop_remainder=True)
    return dataset, dataset.get_dataset_size()


def to_display(img, norm):
    """把网络输出还原成 [0,1] 供 imshow 使用"""
    if norm == "pm1":
        img = (img + 1.0) / 2.0
    return np.clip(img, 0, 1)


# ============================ 谱归一化 ============================
class SNConv2d(nn.Cell):
    """带谱归一化的卷积层（Miyato 等 2018, Spectral Normalization for GANs）。

    做法：把卷积核摊平成矩阵 W (cout × cin·k·k)，用幂迭代估计其最大奇异值 sigma，
    前向时用 W / sigma 做卷积。sigma 每次前向都用上一次的 u 向量再迭代一步，
    训练过程中逐渐收敛到真实值。u 是不参与梯度的持久状态。
    MindSpore 没有现成的 SpectralNorm 层，这里自己实现。
    """

    def __init__(self, cin, cout, kernel, stride, pad_mode, padding=0,
                 weight_init="normal", n_power=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel, stride, pad_mode, padding,
                              weight_init=weight_init, has_bias=False)
        self.stride, self.pad_mode, self.padding = stride, pad_mode, padding
        self.cout = cout
        self.n_power = n_power
        u = np.random.randn(1, cout).astype("float32")
        u /= np.linalg.norm(u) + 1e-12
        self.u = ms.Parameter(ms.Tensor(u), name="sn_u", requires_grad=False)
        self.eps = 1e-12

    def _l2n(self, v):
        return v / (ops.sqrt(ops.reduce_sum(v * v)) + self.eps)

    def construct(self, x):
        w = self.conv.weight
        w_mat = w.reshape(self.cout, -1)                     # (cout, cin*k*k)
        u = self.u
        v = None
        for _ in range(self.n_power):
            v = self._l2n(ops.matmul(u, w_mat))              # (1, cin*k*k)
            u = self._l2n(ops.matmul(v, w_mat.T))            # (1, cout)
        u = ops.stop_gradient(u)
        v = ops.stop_gradient(v)
        sigma = ops.matmul(ops.matmul(u, w_mat), v.T)        # (1,1) 最大奇异值估计
        if self.training:
            ops.assign(self.u, u)                            # 持久化 u，下次接着迭代
        w_sn = w / sigma.reshape(1, 1, 1, 1)
        return ops.conv2d(x, w_sn, stride=self.stride,
                          pad_mode=self.pad_mode, padding=self.padding)


# ============================ 网络 ============================
def build_nets(args):
    wi = Normal(mean=0, sigma=0.02)
    gi = Normal(mean=1, sigma=0.02)
    nz, ngf, ndf, nc = args.nz, args.ngf, args.ndf, 3
    use_sn = args.sn
    use_bn_d = (not args.sn) or args.sn_keep_bn

    def conv_d(cin, cout, k, s, pad_mode, padding=0):
        if use_sn:
            return SNConv2d(cin, cout, k, s, pad_mode, padding, weight_init=wi)
        return nn.Conv2d(cin, cout, k, s, pad_mode, padding, weight_init=wi)

    def bn_d(c):
        return nn.BatchNorm2d(c, gamma_init=gi) if use_bn_d else nn.Identity()

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

    class Discriminator(nn.Cell):
        """use_sigmoid=False 时输出 logits，配 BCEWithLogitsLoss 使用"""
        def __init__(self, use_sigmoid=True):
            super().__init__()
            self.use_sigmoid = use_sigmoid
            self.discriminator = nn.SequentialCell(
                conv_d(nc, ndf, 4, 2, "pad", 1),
                nn.LeakyReLU(0.2),
                conv_d(ndf, ndf * 2, 4, 2, "pad", 1),
                bn_d(ndf * 2), nn.LeakyReLU(0.2),
                conv_d(ndf * 2, ndf * 4, 4, 2, "pad", 1),
                bn_d(ndf * 4), nn.LeakyReLU(0.2),
                conv_d(ndf * 4, ndf * 8, 4, 2, "pad", 1),
                bn_d(ndf * 8), nn.LeakyReLU(0.2),
                conv_d(ndf * 8, 1, 4, 1, "valid"))
            self.sigmoid = nn.Sigmoid()

        def construct(self, x):
            out = self.discriminator(x).reshape(x.shape[0], -1)
            if self.use_sigmoid:
                return self.sigmoid(out)
            return out

    return Generator(), Discriminator(use_sigmoid=not args.logits)


# ============================ 生成器 EMA ============================
def build_ema(generator, args):
    """另建一份结构相同的 G，权重初始化为当前 G 的拷贝；返回 (g_ema, 更新函数)。

    每步 G 更新后调用一次：ema = decay * ema + (1 - decay) * g
    BatchNorm 的滑动均值/方差也一起做 EMA，这样 g_ema 在 eval 模式下可以直接出图。
    """
    g_ema, _ = build_nets(args)
    g_ema.update_parameters_name("ema.")     # 静态图模式要求参数名全局唯一
    ema_params = list(g_ema.get_parameters())
    src_params = list(generator.get_parameters())
    assert len(ema_params) == len(src_params)
    for pe, p in zip(ema_params, src_params):
        pe.set_data(p.data.copy())
    g_ema.set_train(False)

    decay = ms.Tensor(args.ema, ms.float32)
    one_minus = ms.Tensor(1.0 - args.ema, ms.float32)

    def ema_update():
        for pe, p in zip(ema_params, src_params):
            ops.assign(pe, pe * decay + p * one_minus)
        return True

    return g_ema, ema_update


def save_ema_checkpoint(g_ema, path):
    """存盘时去掉 'ema.' 前缀，让 generator_ema.ckpt 能像 generator.ckpt 一样被普通 G 直接加载"""
    plist = [{"name": p.name.replace("ema.", "", 1), "data": p.data} for p in g_ema.get_parameters()]
    try:
        ms.save_checkpoint(plist, path)
    except PermissionError:
        import shutil, tempfile
        tmp = os.path.join(tempfile.gettempdir(), "_ms_" + os.path.basename(path))
        ms.save_checkpoint(plist, tmp)
        shutil.copyfile(tmp, path)
        os.remove(tmp)


def load_ema_checkpoint(g_ema, path):
    params = ms.load_checkpoint(path)
    params = {"ema." + k: v for k, v in params.items()}
    ms.load_param_into_net(g_ema, params)


# ============================ 存盘 ============================
def safe_save_checkpoint(net, path):
    """WSL 下 /mnt/ 挂载的 Windows 盘不支持 chmod，MindSpore 存完 ckpt 会 chmod 400 而报
    PermissionError。这里兜一层：失败就先存到本地临时目录再拷过去（copyfile 不带权限位）。"""
    try:
        ms.save_checkpoint(net, path)
    except PermissionError:
        import shutil, tempfile
        tmp = os.path.join(tempfile.gettempdir(), "_ms_" + os.path.basename(path))
        ms.save_checkpoint(net, tmp)
        shutil.copyfile(tmp, path)
        os.remove(tmp)


# ============================ 绘图 ============================
def save_grid(imgs, path, rows=3, cols=8, norm="01"):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5), dpi=100)
    for ax in np.atleast_1d(axes).flat:
        ax.axis("off")
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if idx < len(imgs):
                axes[r, c].imshow(to_display(imgs[idx], norm))
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_curves(csv_path, out_dir, tag):
    """画两张图：loss 曲线 + D 输出曲线"""
    ep, gl, dl, dx, dgz = [], [], [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            ep.append(int(row["epoch"]))
            gl.append(float(row["g_loss"])); dl.append(float(row["d_loss"]))
            dx.append(float(row["D_x"]));    dgz.append(float(row["D_G_z"]))
    if not ep:
        return

    plt.figure(figsize=(10, 4.5))
    plt.plot(ep, gl, label="Generator loss")
    plt.plot(ep, dl, label="Discriminator loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    plt.title(f"Loss per epoch  [{tag}]"); plt.grid(alpha=.3)
    plt.savefig(os.path.join(out_dir, "curve_loss.png"), bbox_inches="tight")
    plt.close()

    # D 的输出曲线：理想的博弈平衡是两条线都向 0.5 靠拢
    plt.figure(figsize=(10, 4.5))
    plt.plot(ep, dx, label="D(x)  - on real images")
    plt.plot(ep, dgz, label="D(G(z))  - on fake images")
    plt.axhline(0.5, ls="--", c="gray", lw=1, label="equilibrium 0.5")
    plt.ylim(0, 1); plt.xlabel("epoch"); plt.ylabel("mean output"); plt.legend()
    plt.title(f"Discriminator outputs  [{tag}]"); plt.grid(alpha=.3)
    plt.savefig(os.path.join(out_dir, "curve_d_output.png"), bbox_inches="tight")
    plt.close()


def save_interpolation(generator, args, out_dir, steps=8, rows=4):
    """隐空间插值：每行是两个随机向量之间的线性过渡。过渡平滑说明学到的是连续流形"""
    generator.set_train(False)
    rng = np.random.RandomState(args.seed + 999)
    zs = []
    for _ in range(rows):
        z0 = rng.randn(args.nz, 1, 1).astype("float32")
        z1 = rng.randn(args.nz, 1, 1).astype("float32")
        for t in np.linspace(0, 1, steps):
            zs.append((1 - t) * z0 + t * z1)
    z = ms.Tensor(np.stack(zs))
    imgs = generator(z).transpose(0, 2, 3, 1).asnumpy()
    save_grid(imgs, os.path.join(out_dir, "latent_interpolation.png"),
              rows=rows, cols=steps, norm=args.norm)


# ============================ 主流程 ============================
def main():
    args = get_args()
    out_dir = os.path.expanduser(args.out)
    os.makedirs(out_dir, exist_ok=True)

    # 种子：不设种子的话所有对比实验都可以被质疑成随机波动
    ms.set_seed(args.seed)
    np.random.seed(args.seed)

    mode = ms.GRAPH_MODE if args.graph_mode else ms.PYNATIVE_MODE
    try:
        ms.set_context(mode=mode)
        ms.set_device(args.device)
    except AttributeError:                       # 老版本没有 set_device
        ms.set_context(mode=mode, device_target=args.device)

    if out_dir.startswith("/mnt/"):
        print("[提示] 输出目录在 Windows 挂载盘上，每轮写 checkpoint 会明显变慢。\n"
              "       建议改成 WSL 本地路径（如 ~/runs/xxx），训练完再 cp -r 回 D 盘。")

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print("=" * 66)
    print(f"实验 [{args.tag}]  设备 {args.device}  种子 {args.seed}")
    print(f"  归一化 {args.norm} | 翻转增强 {args.flip} | 标签平滑 {args.label_smooth} "
          f"| logits {args.logits}")
    print(f"  谱归一化 {args.sn} (D 保留 BN: {args.sn_keep_bn if args.sn else 'n/a'}) "
          f"| G-EMA {args.ema if args.ema > 0 else '关'} | D lr {args.d_lr or args.lr}")
    print("=" * 66)

    dataset, total = build_dataset(args)
    print(f"数据集: {os.path.expanduser(args.data)}  ->  {total} batch/轮 "
          f"(batch_size={args.batch_size})")

    # 存一张真实样本图，方便和生成结果并排比较
    sample = next(dataset.create_tuple_iterator(output_numpy=True))[0]
    save_grid(sample.transpose(0, 2, 3, 1)[:24],
              os.path.join(out_dir, "sample_real.png"), norm=args.norm)

    generator, discriminator = build_nets(args)
    g_ema, ema_update = (build_ema(generator, args) if args.ema > 0 else (None, None))
    sampler = g_ema if g_ema is not None else generator      # 出图用哪份权重

    d_lr = args.d_lr if args.d_lr is not None else args.lr
    optimizer_G = nn.Adam(generator.trainable_params(), learning_rate=args.lr, beta1=args.beta1)
    optimizer_D = nn.Adam(discriminator.trainable_params(), learning_rate=d_lr, beta1=args.beta1)
    optimizer_G.update_parameters_name("optim_g.")
    optimizer_D.update_parameters_name("optim_d.")

    loss_fn = nn.BCEWithLogitsLoss(reduction="mean") if args.logits \
        else nn.BCELoss(reduction="mean")

    start_epoch = 0
    state_path = os.path.join(out_dir, "state.json")
    if args.resume and os.path.exists(state_path):
        ms.load_param_into_net(generator, ms.load_checkpoint(os.path.join(out_dir, "generator.ckpt")))
        ms.load_param_into_net(discriminator, ms.load_checkpoint(os.path.join(out_dir, "discriminator.ckpt")))
        for name, opt in [("optim_g", optimizer_G), ("optim_d", optimizer_D)]:
            p = os.path.join(out_dir, f"{name}.ckpt")
            if os.path.exists(p):
                ms.load_param_into_net(opt, ms.load_checkpoint(p))
        ema_path = os.path.join(out_dir, "generator_ema.ckpt")
        if g_ema is not None and os.path.exists(ema_path):
            load_ema_checkpoint(g_ema, ema_path)
        start_epoch = json.load(open(state_path))["epoch"]
        print(f"断点续训：从第 {start_epoch + 1} 轮继续")

    # --- 前向 ---
    def generator_forward(valid, bs):
        z = ops.standard_normal((bs, args.nz, 1, 1))
        gen_imgs = generator(z)
        g_loss = loss_fn(discriminator(gen_imgs), valid)
        return g_loss, gen_imgs

    def discriminator_forward(real_imgs, gen_imgs, valid_d, fake):
        d_real = discriminator(real_imgs)
        d_fake = discriminator(gen_imgs)
        d_loss = (loss_fn(d_real, valid_d) + loss_fn(d_fake, fake)) / 2
        if args.logits:                          # 统计量统一成概率，两种模式可比
            d_real = ops.sigmoid(d_real)
            d_fake = ops.sigmoid(d_fake)
        return d_loss, d_real.mean(), d_fake.mean()

    grad_g = ms.value_and_grad(generator_forward, None, optimizer_G.parameters, has_aux=True)
    grad_d = ms.value_and_grad(discriminator_forward, None, optimizer_D.parameters, has_aux=True)

    def train_step(imgs):
        bs = imgs.shape[0]
        valid_g = ops.ones((bs, 1), ms.float32)                       # G 想让 D 判成真，目标恒为 1
        valid_d = ops.ones((bs, 1), ms.float32) * args.label_smooth   # 单边标签平滑只作用于 D
        fake = ops.zeros((bs, 1), ms.float32)

        (g_loss, gen_imgs), g_grads = grad_g(valid_g, bs)
        optimizer_G(g_grads)
        if ema_update is not None:
            ema_update()
        (d_loss, d_x, d_gz), d_grads = grad_d(imgs, gen_imgs, valid_d, fake)
        optimizer_D(d_grads)
        return g_loss, d_loss, d_x, d_gz

    if args.graph_mode:
        train_step = ms.jit(train_step)

    # 固定噪声只采样一次 —— 这样每轮生成图对应同一批隐向量，逐轮对比才有意义
    fixed_noise = ms.Tensor(
        np.random.RandomState(args.seed).randn(24, args.nz, 1, 1).astype("float32"))

    csv_path = os.path.join(out_dir, "metrics.csv")
    if not os.path.exists(csv_path) or start_epoch == 0:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "g_loss", "d_loss", "D_x", "D_G_z", "sec"])

    print(f"\n开始训练 {args.epochs} 轮\n" + "-" * 66)
    for epoch in range(start_epoch, args.epochs):
        generator.set_train(); discriminator.set_train()
        t0 = time.time()
        acc = np.zeros(4)
        n = 0
        for i, (imgs,) in enumerate(dataset.create_tuple_iterator()):
            g_loss, d_loss, d_x, d_gz = train_step(imgs)
            acc += np.array([float(g_loss.asnumpy()), float(d_loss.asnumpy()),
                             float(d_x.asnumpy()), float(d_gz.asnumpy())])
            n += 1
            if i % 100 == 0:
                print(f"  [{epoch+1:2d}/{args.epochs}][{i+1:4d}/{total}] "
                      f"Loss_D {float(d_loss.asnumpy()):7.4f}  Loss_G {float(g_loss.asnumpy()):7.4f}  "
                      f"D(x) {float(d_x.asnumpy()):.3f}  D(G(z)) {float(d_gz.asnumpy()):.3f}")
        m = acc / max(n, 1)
        sec = time.time() - t0

        # 每轮都落盘，崩溃时不会像原版那样丢掉全部训练数据
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{m[0]:.6f}", f"{m[1]:.6f}",
                                    f"{m[2]:.6f}", f"{m[3]:.6f}", f"{sec:.1f}"])

        if (epoch + 1) % args.sample_every == 0 or epoch + 1 == args.epochs:
            sampler.set_train(False)
            imgs = sampler(fixed_noise).transpose(0, 2, 3, 1).asnumpy()
            save_grid(imgs, os.path.join(out_dir, f"generated_epoch{epoch+1}.png"),
                      norm=args.norm)
            if g_ema is not None:                # 另存一份原始权重的图，方便对比 EMA 的效果
                generator.set_train(False)
                imgs = generator(fixed_noise).transpose(0, 2, 3, 1).asnumpy()
                save_grid(imgs, os.path.join(out_dir, f"generated_raw_epoch{epoch+1}.png"),
                          norm=args.norm)

        if g_ema is not None:
            save_ema_checkpoint(g_ema, os.path.join(out_dir, "generator_ema.ckpt"))
        safe_save_checkpoint(generator, os.path.join(out_dir, "generator.ckpt"))
        safe_save_checkpoint(discriminator, os.path.join(out_dir, "discriminator.ckpt"))
        safe_save_checkpoint(optimizer_G, os.path.join(out_dir, "optim_g.ckpt"))
        safe_save_checkpoint(optimizer_D, os.path.join(out_dir, "optim_d.ckpt"))
        json.dump({"epoch": epoch + 1}, open(state_path, "w"))

        print(f"--- 第 {epoch+1}/{args.epochs} 轮完成  用时 {sec:.0f}s  "
              f"G {m[0]:.3f}  D {m[1]:.3f}  D(x) {m[2]:.3f}  D(G(z)) {m[3]:.3f} ---\n")

    save_curves(csv_path, out_dir, args.tag)
    save_interpolation(sampler, args, out_dir)

    print("=" * 66)
    print(f"训练完成，结果在 {out_dir}")
    print("  metrics.csv                所有轮次的指标，可直接导入表格做对比")
    print("  curve_loss.png             loss 曲线")
    print("  curve_d_output.png         D(x) / D(G(z)) 曲线 —— 看博弈是否平衡")
    print("  latent_interpolation.png   隐空间插值图")
    print("  generated_epoch*.png       固定噪声下的逐轮生成结果")


if __name__ == "__main__":
    main()
