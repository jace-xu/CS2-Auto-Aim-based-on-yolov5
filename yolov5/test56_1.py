import torch
import cv2
import numpy as np
import mss
import time
import sys
import os

# 添加 YOLOv5 模型代码路径
FILE = os.path.abspath(__file__)
ROOT = os.path.dirname(FILE)
sys.path.append(ROOT)

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# 设置模型权重路径
weights = r"D:\Python files\cs2-autoaim\9may_yolov5x.pt"

# 加载模型
device = select_device('')
model = attempt_load(weights, device=device)  # 修改这里，将 map_location 改为 device
model.float().fuse().eval() # 确保模型进入评估模式并进行融合（如果适用）

# 获取类别名
names = model.names if hasattr(model, 'names') else model.module.names if hasattr(model, 'module') else model.model.names # 兼容不同保存方式的names获取

# 初始化 MSS 截屏（监视器1 = 全屏）
sct = mss.mss()
monitor = sct.monitors[1]

print("🔍 正在检测 CT 目标数，按 q 退出...")

while True:
    start_time = time.time()

    # 截取屏幕并转为 BGR 格式
    screenshot = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    # 预处理图像
    img = cv2.resize(frame, (640, 640))
    img_tensor = torch.from_numpy(img).to(device).float()
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    img_tensor /= 255.0

    # 模型推理
    pred = model(img_tensor)[0]
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    # 统计 T 数量并绘制检测框
    t_count = 0
    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                class_name = names[int(cls)]
                if class_name == "T":
                    t_count += 1
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 红框
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 0, 255), 2)

    # 显示数量和 FPS
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f"T Targets: {t_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

    # 展示图像
    cv2.imshow("CT Detector - Full Screen", frame)

    # 按 q 退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
