import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import ultralytics
from ultralytics import YOLO
import torch
import multiprocessing
# from ultralytics.nn.modules import Block, Conv
from torch import nn

# 检查环境和设备
print(ultralytics.checks())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
data_yaml_path = "/root/YoloRecognition/dataset/Helment/data.yaml"

# class CBAM(Block):
#     """ Convolutional Block Attention Module (CBAM) - https://arxiv.org/abs/1807.06521 """
#     def __init__(self, c1, reduction=16, kernel_size=7):
#         super().__init__()
#         self.channel_attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             Conv(c1, c1 // reduction, 1, act=nn.ReLU()),
#             Conv(c1 // reduction, c1, 1, act=nn.Sigmoid())
#         )
#         self.spatial_attention = nn.Sequential(
#             Conv(2, 1, kernel_size, padding=kernel_size//2, act=nn.Sigmoid())
#         )

#     def forward(self, x):
#         ca = self.channel_attention(x) * x
#         sa_input = torch.cat([ca.mean(dim=1, keepdim=True), ca.max(dim=1, keepdim=True)[0]], dim=1)
#         sa = self.spatial_attention(sa_input)
#         return sa * ca

# # 将自定义模块添加到Ultralytics的模块字典中
# torch.nn.Module.dump_patches = True # 有时需要这个来避免保存错误
# setattr(ultralytics.nn.modules, 'CBAM', CBAM) # 关键！注册CBAM模块

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # 加载预训练模型
    model = YOLO('yolov8n.pt')

    # 训练，添加调试参数
    results = model.train(
        data=data_yaml_path,
        epochs=100,
        imgsz=640,
        batch=16,
        name='helmet_cbam_training_8v2',
        device=device,
        patience=10,
        save=True,
        plots=True,
        workers=4,  # 保持0，成功后可调到4
        rect=True,  # 矩形训练，减少内存需求
        cache=False,  # 禁用缓存
        exist_ok=True,  # 允许覆盖现有runs目录
        hsv_h=0.015, # 色调 (Hue)
        hsv_s=0.7,   # 饱和度 (Saturation) - 调高
        hsv_v=0.4,   # 明度 (Value)
        translate=0.2, # 平移 - 调高
        scale=0.9,     # 缩放 - 调高
        fliplr=0.5,    # 水平翻转
        mosaic=1.0,    # 确保mosaic开启 (1.0是100%概率)
        mixup=0.2,     # 确保mixup开启 (0.2是20%概率)
    )

    # 评估
    metrics = model.val()
    print(f"mAP@0.5: {metrics.box.map50}")

    # 保存模型
    model.save('helmet_best.pt')
    print("训练完成！模型保存为 helmet_best.pt")
