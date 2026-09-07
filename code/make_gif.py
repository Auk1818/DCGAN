"""生成DCGAN训练过程GIF动画"""
import os
import numpy as np
from PIL import Image

output_dir = r'C:\dcgan_result'
result_dir = r'C:\Users\lenovo\Desktop\机器视觉实验\result'
num_epochs = 40

print("正在生成GIF动画...")
frames = []
for epoch in range(1, num_epochs + 1):
    img_path = os.path.join(output_dir, f'generated_epoch{epoch}.png')
    if os.path.exists(img_path):
        img = Image.open(img_path)
        frames.append(img)
        print(f"  加载 Epoch {epoch}/{num_epochs}")

gif_path = os.path.join(result_dir, 'dcgan.gif')
frames[0].save(
    gif_path,
    save_all=True,
    append_images=frames[1:],
    duration=1000,
    loop=0
)
print(f"GIF已保存: {gif_path}")
print(f"共 {len(frames)} 帧")
