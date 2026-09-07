# 基于 MindSpore 的 DCGAN 生成漫画头像

> 机器视觉实验 | MindSpore · DCGAN · matplotlib · numpy

## 项目简介

使用 **DCGAN（深度卷积生成对抗网络）** 在 MindSpore 框架下训练漫画头像生成模型。通过 Generator 和 Discriminator 的对抗训练，让模型学会从随机噪声中生成 64×64 的漫画风格人脸图像。

- **数据集**：漫画头像图片（通过 `download/` 下载）
- **框架**：MindSpore（CPU 模式）
- **训练结果**：完成 40 轮训练，生成效果随 epoch 逐步提升

---

## 目录结构

```
机器视觉实验/
├── 基于MindSpore的DCGAN生成漫画头像实验手册.docx   # 实验指导手册
├── 华为云ModelArts.docx                          # ModelArts 云平台说明
├── README.md                                     # 项目说明文档
├── ai-log.md                                     # AI 协作记录
│
├── code/                                         # 源代码
│   ├── train_dcgan.py                            # DCGAN 训练脚本（主程序）
│   ├── monitor.py                                # 训练实时监控服务器
│   ├── training_demo.html                        # 训练过程动态演示页面
│   └── make_gif.py                               # 训练过程GIF动画生成脚本
│
├── download/                                     # 数据集下载目录
│
└── result/                                       # 训练结果输出
    ├── sample_data.png                           # 训练数据样本可视化
    ├── generated_epoch{1-40}.png                 # 每轮生成图片
    ├── loss_curve_epoch{1-40}.png                # 每轮 Loss 曲线
    ├── dcgan.gif                                 # 训练过程动画（40帧）
    ├── generator.ckpt                            # 生成器模型权重
    └── discriminator.ckpt                        # 判别器模型权重
```

---

## 快速开始

### 环境要求

```bash
# 使用 conda 创建 Python 3.10 环境
conda create -n python310 python=3.10
conda activate python310

# 安装依赖
pip install mindspore matplotlib numpy
```

### 运行训练

```bash
python code/train_dcgan.py
```

训练过程中每轮会自动保存：
- 生成图片：`result/generated_epoch{N}.png`
- Loss 曲线：`result/loss_curve_epoch{N}.png`
- 模型权重：`result/generator.ckpt`、`result/discriminator.ckpt`

### 训练监控

训练启动后，可同时运行监控服务器实时查看进度：

```bash
# 默认端口 8765
python code/monitor.py

# 自定义端口和结果目录
python code/monitor.py --port 8080 --dir C:\dcgan_result
```

浏览器访问 `http://localhost:8765`，页面每 8 秒自动刷新，显示：
- 训练进度条（当前 epoch / 总 epoch）
- 最新生成图片
- 最新 Loss 曲线

### 训练过程动态演示

```bash
# 直接用浏览器打开即可（无需启动服务器）
code/training_demo.html
```

页面加载后显示 Epoch 1 的生成结果，点击 **播放** 按钮开始自动轮播，也可拖动滑块或使用键盘左右箭头手动翻页。支持三种播放速度（慢/中/快）。

---

## 模型参数

| 参数 | 值 | 说明 |
|------|------|------|
| `batch_size` | 128 | 批次大小 |
| `image_size` | 64 | 生成图片尺寸 64×64 |
| `nz` | 100 | 隐空间向量维度 |
| `ngf` | 64 | 生成器特征图数量 |
| `ndf` | 64 | 判别器特征图数量 |
| `lr` | 0.0002 | 学习率 |
| `beta1` | 0.5 | Adam 优化器参数 |
| `num_epochs` | 40 | 训练轮数 |

---

## 训练结果

训练完成 40 轮后，生成效果从随机噪声逐步演化为可辨识的漫画人脸。

### Loss 曲线趋势

- **Generator Loss**：从较高值逐步下降并趋于稳定
- **Discriminator Loss**：与 Generator 形成对抗平衡
- 两个 Loss 最终在较低水平震荡，说明 GAN 达到了较好的平衡状态

### 生成效果

| 阶段 | 说明 |
|------|------|
| Epoch 1-5 | 随机噪声，无明显人脸特征 |
| Epoch 10-20 | 出现模糊的人脸轮廓 |
| Epoch 20-30 | 五官逐渐清晰 |
| Epoch 30-40 | 生成质量较好，具备漫画风格特征 |

---

## 技术栈

| 技术 | 用途 |
|------|------|
| Python 3.10 | 主语言 |
| MindSpore | 深度学习框架（CPU 模式） |
| numpy | 数值计算 |
| matplotlib | 可视化生成图片和 Loss 曲线 |

---

## 提交清单

| 文件 | 说明 |
|------|------|
| `README.md` | 项目说明文档 |
| `ai-log.md` | AI 协作记录 |
| `code/train_dcgan.py` | DCGAN 训练脚本 |
| `code/monitor.py` | 训练监控服务器 |
| `code/training_demo.html` | 训练过程动态演示页面（浏览器打开即可查看） |
| `code/make_gif.py` | 训练过程GIF动画生成脚本 |
| `result/` | 训练结果（生成图片 + Loss 曲线 + GIF动画 + 模型权重） |
