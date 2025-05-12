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
weights = os.path.join(ROOT, "runs", "train", "exp7", "weights", "best.pt")

# 加载模型（兼容新版YOLOv5）
device = select_device('')
model_data = torch.load(weights, map_location=device)
model = model_data['model'].float().fuse().eval()

# 获取类别名
names = model.names

# 初始化 MSS 截屏
sct = mss.mss()
monitor = sct.monitors[1]  # 全屏

print("🔍 正在检测 CT 目标数，按 q 退出...")

while True:
    start_time = time.time()

    # 屏幕截图并转换为 BGR 图像
    screenshot = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    # 缩放为模型输入大小
    img = cv2.resize(frame, (640, 640))
    img_tensor = torch.from_numpy(img).to(device).float()
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # BxCxHxW
    img_tensor /= 255.0

    # 推理
    pred = model(img_tensor)[0]
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    # 统计 CT 数量
    ct_count = 0
    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                if names[int(cls)] == "CT":
                    ct_count += 1

    # 显示检测数量和 FPS
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f"CT Targets: {ct_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

    # 显示窗口
    cv2.imshow("CT Detector - Full Screen", frame)

    # 按 q 退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
