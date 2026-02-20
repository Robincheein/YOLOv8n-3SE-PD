import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import ultralytics
from ultralytics import YOLO
import torch
import multiprocessing
from torch import nn

# 检查环境和设备
print(ultralytics.checks())
device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")

# 调试：检查数据集路径
data_yaml_path = "/root/YoloRecognition/dataset/Helment/data.yaml"
# print(f"Data YAML path exists: {os.path.exists(data_yaml_path)}")

# 手动注册SE模块
import ultralytics.nn.tasks
from ultralytics.nn.modules import SE

ultralytics.nn.tasks.SE = SE

if __name__ == "__main__":
    multiprocessing.freeze_support()

    try:
        # 加载自定义SE模型 - 这里可能会出错
        custom_yaml_path = "yolov8n_se_custom.yaml"
        print(f"加载自定义YAML: {os.path.exists(custom_yaml_path)}")

        # 添加详细的错误处理
        try:
            model = YOLO(custom_yaml_path)
            print("✅ YAML解析成功！")
        except Exception as e:
            print(f"❌ YAML解析错误: {e}")
            raise e

        # 如果YAML解析成功，继续调试
        # print("模型结构摘要:")
        # print(model.model)

        # 验证模型结构中是否包含SE模块
        # se_modules = []
        # for name, module in model.model.named_modules():
        #     if isinstance(module, SE):
        #         se_modules.append(name)
        #         print(f"找到SE模块: {name}")

        # if not se_modules:
        #     print("⚠️ 警告: 未找到SE模块！")
        # else:
        #     print(f"✅ 找到 {len(se_modules)} 个SE模块")
        # 冻结部分层，只训练特定层
        for name, param in model.model.named_parameters():
            if 'SE' not in name:  # 只训练SE模块和新添加的层
                param.requires_grad = False

        class_weights = {
            "with helmet": 1.5,    # 增加权重
            "without helmet": 2.0, # 这个类别需要更多关注
            "rider": 1.0,
            "number plate": 1.0
        }
        
        # 训练参数
        results = model.train(
            data=data_yaml_path,
            epochs=500,  # 增加训练轮次
            imgsz=640,
            batch=24,
            name="helmet_se_custom_v3",
            device=device,
            patience=15,
            save=True,
            plots=True,
            workers=0,  # 可适当增加
            rect=True,
            cache=False,
            exist_ok=True,
            # 数据增强
            # hsv_h=0.015,
            # hsv_s=0.7,
            # hsv_v=0.4,
            # translate=0.1,  # 降低平移
            # scale=0.5,  # 降低缩放
            # fliplr=0.5,
            # mosaic=1.0,
            # mixup=0.1,  # 降低 mixup
            # # 优化器
            # optimizer="AdamW",
            # lr0=0.001,  # 降低学习率
            # momentum=0.05,
            # weight_decay=0.001,
            # warmup_epochs=3,
            # close_mosaic=10,
            # 添加类别权重（如果类别不平衡）
            # cls=1.5,       # 可尝试
            # box=7.5,       # 可尝试
        )

        # 验证模型
        metrics = model.val()
        print(f"验证集mAP@0.5: {metrics.box.map50:.4f}")

        # 保存最终模型
        model.save("helmet_best_se.pt")
        print("训练完成！模型保存为 helmet_best_se.pt")

    except Exception as e:
        print(f"❌ 错误: {e}")