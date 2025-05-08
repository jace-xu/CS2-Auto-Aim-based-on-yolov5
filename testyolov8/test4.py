from ultralytics import YOLO
import cv2
import numpy as np
import mss
import time
import pyautogui
import keyboard

# 加载 YOLOv8 模型
model = YOLO("yolov8n.pt")

# 获取屏幕信息
sct = mss.mss()
screen = sct.monitors[1]
screen_width = screen["width"]
screen_height = screen["height"]

# 设置中心 500x500 区域
monitor = {
    "top": (screen_height - 500) // 2,
    "left": (screen_width - 500) // 2,
    "width": 500,
    "height": 500
}

# 中心坐标（准星位置）
center_x = 500 // 2
center_y = 500 // 2

# 自瞄状态开关
aimbot_enabled = False

# 记录时间，计算FPS
prev_time = time.time()

print("⚙️ 按下 F1 开/关自瞄功能，按 q 退出程序")

while True:
    # 记录开始时间
    start_time = time.time()

    # 检查是否切换自瞄
    if keyboard.is_pressed('f1'):
        aimbot_enabled = not aimbot_enabled
        print(f"[切换] 自瞄状态: {'✅ 开启' if aimbot_enabled else '❌ 关闭'}")
        time.sleep(0.5)  # 防止多次触发

    # 截图屏幕中心区域
    screenshot = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    # 运行 YOLOv8 检测
    results = model(frame)
    result = results[0]

    person_count = 0
    target_found = False

    # 遍历检测到的框
    for box in result.boxes:
        cls_id = int(box.cls[0])
        if cls_id == 0:  # 如果是person
            person_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 计算人物框的中心点
            target_x = (x1 + x2) // 2
            target_y = (y1 + y2) // 2

            # 绘制识别框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 只对第一个目标自瞄
            if not target_found and aimbot_enabled:
                # 计算目标的偏移量
                dx = target_x - center_x
                dy = target_y - center_y

                # 限制偏移，避免过大的鼠标跳跃
                if abs(dx) < 200 and abs(dy) < 200:
                    # 快速移动鼠标到目标的中心
                    pyautogui.moveTo(target_x + (screen_width - 500) // 2, 
                                     target_y + (screen_height - 500) // 2, duration=0.1)  # 快速移动到目标位置

                    # 标记为已找到目标
                    target_found = True

    # 计算FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)

    # 显示人数和FPS
    cv2.putText(frame, f"Persons: {person_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # 显示窗口
    cv2.imshow("YOLO Aimbot Demo", frame)

    # 按下 q 退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

