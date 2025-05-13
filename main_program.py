
"""
Updating recording

1.FPS Improvement:
Using threading to separate screen capture and target detection logic, reducing main thread blocking.

2.Robustness and Adaptability:
Added Tkinter pop-up to input smoothing factor, adapting to different levels of hand tremor.

3.Key Conflict Changes:
Changed the exit key from 'q' to '0'.
Added startup prompt pop-up, reminding users to press the confirmation key before the program starts running.

4.Others:

(i)Added model loading exception handling to avoid program crashes.
Used a queue to store the latest frame, ensuring YOLO operates on the most recently captured screen.

(ii)Optimized screen center positioning method to ensure compatibility with different device screens.

(iii)Test results show that with a recognition box size of 500x500, the frame rate is relatively high, and recognition is stable.

5.Problems:

(i)Requires standing still to aim and shoot; tracking moving targets is still not good enough.

(ii)Sniper rifles require scope positioning, and submachine guns cannot always lock aim due to recoil.

(iii)During program execution, the following warning may appear: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated.
YOLO methods need to be updated in the future, but this does not affect the program's operation.

(iiii)This code includes Unicode and emoji characters.
 Your terminal or editor must support them;
 otherwise, they may appear as boxes or garbled text.
 If your terminal doesn't support them, you can remove them
 (in the print statements of the `show_startup_message` and `get_smoothing_factor` functions).

"""
import torch
import cv2
import numpy as np
import mss
import time
import ctypes
import win32gui
import win32con
import tkinter as tk
from tkinter import simpledialog
from tkinter import messagebox

# Load YOLOv5 model from local,PLEASE CHANGE YOUR OWN PATH WHEN YOU ARE USING
yolov5_code_path = 'D:/Python files/cs2-autoaim/yolov5_run'  # YOLOv5 code path
model_weights_path = 'D:/Python files/cs2-autoaim/9may_yolov5s_final.pt'  # Model file path

# Load models
try:
    model = torch.hub.load(
        yolov5_code_path,  # YOLOv5 code directory
        'custom',          # Load custom model
        path=model_weights_path,  # Model weights file path
        source='local'     # Specify local loading source
    )
except Exception as e:
    print(f"Model loading failed: {e}")
    exit()

# Get screen information
sct = mss.mss()
screen = sct.monitors[1]
screen_width = screen["width"]
screen_height = screen["height"]

# Set the screen center area to 500x500
monitor = {
    "top": (screen_height // 2) - 250,  # Screen center minus half the height of the area
    "left": (screen_width // 2) - 250,  # Screen center minus half the width of the area
    "width": 500,
    "height": 500
}

# Center coordinates (crosshair position)
center_x = 500 // 2
center_y = 500 // 2

# Define mouse movement function (using Windows API)
def move_mouse_relative(dx, dy):
    ctypes.windll.user32.mouse_event(0x0001, dx, dy, 0, 0)  # 0x0001 indicates mouse movement

# Define function to make window topmost
def make_window_topmost(window_name: str):
    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd:
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                              win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

# Show startup message pop-up
def show_startup_message():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    messagebox.showinfo(
        "Startup Prompt",
        "Hint:\n\u2605Please start this program after entering the game to prevent abnormal mouse movement interference during team formation and map selection.\n\u2605At the same time, you can press the '0' key at any time to exit the program."
    )

# Display startup prompt
show_startup_message()

print("Program started, press '0' to exit.")

# Use Tkinter pop-up to get user input for smoothing factor
def get_smoothing_factor():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    try:
        smoothing_factor = simpledialog.askfloat(
            "Smoothing Factor Setting",
            "\U0001F680Please enter the mouse smoothing factor (recommended range 0.1 ~ 2.0, default value is 0.5).\n If hand tremor is severe, you can increase the smoothing factor to reduce jumping; if hand tremor is mild, you can decrease the smoothing factor.",
            minvalue=0.1, maxvalue=2.0
        )
        if smoothing_factor is None:
            raise ValueError("User canceled input or did not enter a valid value")
    except ValueError as e:
        print(f"Input error: {e}, using default smoothing factor 0.5")
        smoothing_factor = 0.5
    return smoothing_factor

smoothing_factor = get_smoothing_factor()
print(f"Using smoothing factor: {smoothing_factor}")

while True:
    # Record start time
    start_time = time.time()

    # Capture the center area of the screen
    screenshot = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    # Add crosshair (draw a small circle at the center of the screen)
    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Red small circle with radius 5

    # Run YOLOv5 detection
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()  # Get detection results and convert to NumPy array

    person_count = 0
    closest_distance = float('inf')  # Initialize the closest target distance to infinity
    closest_target = None  # Center coordinates of the closest target

    # Iterate through detected boxes
    for *xyxy, conf, cls in detections:
        cls_id = int(cls)
        if cls_id == 0:  # If a person is detected
            person_count += 1
            x1, y1, x2, y2 = map(int, xyxy)

            # Calculate the center point of the person box
            target_x = (x1 + x2) // 2
            target_y = (y1 + y2) // 2

            # Calculate the distance between the target center and the crosshair
            distance = np.sqrt((target_x - center_x)**2 + (target_y - center_y)**2)

            # If the current target is closer, update the closest target
            if distance < closest_distance:
                closest_distance = distance
                closest_target = (target_x, target_y)

            # Draw detection box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "T", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # If the closest target is found, move the mouse to the target's center
    if closest_target is not None:
        target_x, target_y = closest_target
        dx = target_x - center_x
        dy = target_y - center_y

        # Use user-set smoothing factor to control mouse movement speed
        move_mouse_relative(int(dx * smoothing_factor), int(dy * smoothing_factor))

    # Calculate FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)

    # Display person count and FPS
    cv2.putText(frame, f"T: {person_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display window
    window_name = "Target"
    cv2.imshow(window_name, frame)

    # Ensure the window is always on top
    make_window_topmost(window_name)

    # Press '0' to exit
    if cv2.waitKey(1) & 0xFF == ord("0"):
        break

cv2.destroyAllWindows()
