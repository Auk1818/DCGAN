"""基于MindSpore的DCGAN生成漫画头像 - 训练脚本"""

import os
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore import nn, ops
from mindspore.common.initializer import Normal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ======================== 参数设置 ========================
batch_size = 128
image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
num_epochs = 40
lr = 0.0002
beta1 = 0.5

# 路径设置 (使用junction避免中文路径问题)
data_path = r'C:\dcgan_faces'
output_dir = r'C:\dcgan_result'
os.makedirs(output_dir, exist_ok=True)

# 设置MindSpore上下文
ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

# ======================== 数据加载 ========================
def create_dataset_imagenet(dataset_path):
    dataset = ds.ImageFolderDataset(dataset_path,
                                    num_parallel_workers=4,
                                    shuffle=True,
                                    decode=True)
    transforms = [
        vision.Resize(image_size),
        vision.CenterCrop(image_size),
        vision.HWC2CHW(),
        lambda x: ((x / 255).astype("float32"))
    ]
    dataset = dataset.project('image')
    dataset = dataset.map(transforms, 'image')
    dataset = dataset.batch(batch_size)
    return dataset

print("Loading dataset...")
dataset = create_dataset_imagenet(data_path)
total = dataset.get_dataset_size()
print(f"Dataset loaded. Total batches per epoch: {total}")

# 可视化部分训练数据
sample_data = next(dataset.create_tuple_iterator(output_numpy=True))
plt.figure(figsize=(10, 3), dpi=140)
for i, image in enumerate(sample_data[0][:30], 1):
    plt.subplot(3, 10, i)
    plt.axis("off")
    plt.imshow(image.transpose(1, 2, 0))
plt.savefig(os.path.join(output_dir, 'sample_data.png'), bbox_inches='tight')
plt.close()
print("Sample data saved.")

# ======================== 网络定义 ========================
weight_init = Normal(mean=0, sigma=0.02)
gamma_init = Normal(mean=1, sigma=0.02)


class Generator(nn.Cell):
    def __init__(self):
        super(Generator, self).__init__()
        self.generator = nn.SequentialCell(
            nn.Conv2dTranspose(nz, ngf * 8, 4, 1, 'valid', weight_init=weight_init),
            nn.BatchNorm2d(ngf * 8, gamma_init=gamma_init),
            nn.ReLU(),
            nn.Conv2dTranspose(ngf * 8, ngf * 4, 4, 2, 'pad', 1, weight_init=weight_init),
            nn.BatchNorm2d(ngf * 4, gamma_init=gamma_init),
            nn.ReLU(),
            nn.Conv2dTranspose(ngf * 4, ngf * 2, 4, 2, 'pad', 1, weight_init=weight_init),
            nn.BatchNorm2d(ngf * 2, gamma_init=gamma_init),
            nn.ReLU(),
            nn.Conv2dTranspose(ngf * 2, ngf, 4, 2, 'pad', 1, weight_init=weight_init),
            nn.BatchNorm2d(ngf, gamma_init=gamma_init),
            nn.ReLU(),
            nn.Conv2dTranspose(ngf, nc, 4, 2, 'pad', 1, weight_init=weight_init),
            nn.Tanh()
        )

    def construct(self, x):
        return self.generator(x)


class Discriminator(nn.Cell):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.SequentialCell(
            nn.Conv2d(nc, ndf, 4, 2, 'pad', 1, weight_init=weight_init),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 'pad', 1, weight_init=weight_init),
            nn.BatchNorm2d(ndf * 2, gamma_init=gamma_init),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 'pad', 1, weight_init=weight_init),
            nn.BatchNorm2d(ndf * 4, gamma_init=gamma_init),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 'pad', 1, weight_init=weight_init),
            nn.BatchNorm2d(ndf * 8, gamma_init=gamma_init),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 8, 1, 4, 1, 'valid', weight_init=weight_init),
        )
        self.adv_layer = nn.Sigmoid()

    def construct(self, x):
        out = self.discriminator(x)
        out = out.reshape(out.shape[0], -1)
        return self.adv_layer(out)


generator = Generator()
discriminator = Discriminator()

# ======================== 损失函数和优化器 ========================
adversarial_loss = nn.BCELoss(reduction='mean')
optimizer_D = nn.Adam(discriminator.trainable_params(), learning_rate=lr, beta1=beta1)
optimizer_G = nn.Adam(generator.trainable_params(), learning_rate=lr, beta1=beta1)
optimizer_G.update_parameters_name('optim_g.')
optimizer_D.update_parameters_name('optim_d.')

# ======================== 训练函数 ========================
def generator_forward(real_imgs, valid):
    z = ops.standard_normal((real_imgs.shape[0], nz, 1, 1))
    gen_imgs = generator(z)
    g_loss = adversarial_loss(discriminator(gen_imgs), valid)
    return g_loss, gen_imgs


def discriminator_forward(real_imgs, gen_imgs, valid, fake):
    real_loss = adversarial_loss(discriminator(real_imgs), valid)
    fake_loss = adversarial_loss(discriminator(gen_imgs), fake)
    d_loss = (real_loss + fake_loss) / 2
    return d_loss


grad_generator_fn = ms.value_and_grad(generator_forward, None,
                                      optimizer_G.parameters,
                                      has_aux=True)
grad_discriminator_fn = ms.value_and_grad(discriminator_forward, None,
                                          optimizer_D.parameters)


@ms.jit
def train_step(imgs):
    valid = ops.ones((imgs.shape[0], 1), ms.float32)
    fake = ops.zeros((imgs.shape[0], 1), ms.float32)
    (g_loss, gen_imgs), g_grads = grad_generator_fn(imgs, valid)
    optimizer_G(g_grads)
    d_loss, d_grads = grad_discriminator_fn(imgs, gen_imgs, valid, fake)
    optimizer_D(d_grads)
    return g_loss, d_loss, gen_imgs


# ======================== 训练循环 ========================
G_losses = []
D_losses = []
image_list = []

print(f"\nStarting training for {num_epochs} epochs...")
print(f"Dataset: {data_path} ({total} batches, batch_size={batch_size})")
print("=" * 60)

for epoch in range(num_epochs):
    generator.set_train()
    discriminator.set_train()
    for i, (imgs, ) in enumerate(dataset.create_tuple_iterator()):
        g_loss, d_loss, gen_imgs = train_step(imgs)
        if i % 100 == 0 or i == total - 1:
            print('[%2d/%d][%3d/%d]   Loss_D:%7.4f  Loss_G:%7.4f' % (
                epoch + 1, num_epochs, i + 1, total, d_loss.asnumpy(), g_loss.asnumpy()))
        D_losses.append(d_loss.asnumpy())
        G_losses.append(g_loss.asnumpy())

    # 每个epoch结束后生成图片
    generator.set_train(False)
    fixed_noise = ops.standard_normal((batch_size, nz, 1, 1))
    img = generator(fixed_noise)
    image_list.append(img.transpose(0, 2, 3, 1).asnumpy())

    # 保存每轮生成的图片
    gen_img = img.transpose(0, 2, 3, 1).asnumpy()
    fig, axes = plt.subplots(3, 8, figsize=(12, 5), dpi=100)
    for ax in axes.flat:
        ax.axis("off")
    for j in range(3):
        for k in range(8):
            axes[j, k].imshow(np.clip(gen_img[j * 8 + k], 0, 1))
    plt.savefig(os.path.join(output_dir, f'generated_epoch{epoch + 1}.png'), bbox_inches='tight')
    plt.close()

    # 保存模型参数
    ms.save_checkpoint(generator, os.path.join(output_dir, "generator.ckpt"))
    ms.save_checkpoint(discriminator, os.path.join(output_dir, "discriminator.ckpt"))

    # 保存每轮loss曲线
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G", color='blue')
    plt.plot(D_losses, label="D", color='orange')
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'loss_curve_epoch{epoch + 1}.png'), bbox_inches='tight')
    plt.close()

    print(f"--- Epoch {epoch + 1}/{num_epochs} completed ---\n")

# ======================== 结果保存 ========================
# 保存最终loss曲线
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G", color='blue')
plt.plot(D_losses, label="D", color='orange')
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(output_dir, 'loss_curve_final.png'), bbox_inches='tight')
plt.close()

# 保存loss数据
np.savez(os.path.join(output_dir, 'losses.npz'), G_losses=G_losses, D_losses=D_losses)

# 生成GIF动画
show_list = []
fig = plt.figure(figsize=(8, 3), dpi=120)
for epoch_idx in range(len(image_list)):
    images = []
    for i in range(3):
        row = np.concatenate((image_list[epoch_idx][i * 8:(i + 1) * 8]), axis=1)
        images.append(row)
    img = np.clip(np.concatenate((images[:]), axis=0), 0, 1)
    plt.axis("off")
    show_list.append([plt.imshow(img)])
ani = animation.ArtistAnimation(fig, show_list, interval=1000, repeat_delay=1000, blit=True)
ani.save(os.path.join(output_dir, 'dcgan.gif'), writer='pillow', fps=1)
plt.close()

# 最终生成结果展示
ms.load_checkpoint(os.path.join(output_dir, "generator.ckpt"), generator)
fixed_noise = ops.standard_normal((batch_size, nz, 1, 1))
img64 = generator(fixed_noise).transpose(0, 2, 3, 1).asnumpy()
fig = plt.figure(figsize=(8, 3), dpi=120)
images = []
for i in range(3):
    images.append(np.concatenate((img64[i * 8:(i + 1) * 8]), axis=1))
img = np.clip(np.concatenate((images[:]), axis=0), 0, 1)
plt.axis("off")
plt.imshow(img)
plt.savefig(os.path.join(output_dir, 'final_result.png'), bbox_inches='tight', dpi=120)
plt.close()

print("=" * 60)
print("Training completed! Results saved to:", output_dir)
