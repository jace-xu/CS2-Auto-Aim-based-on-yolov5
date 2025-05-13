

# Gaming Meets Rehabilitation: A Fun-Oriented Rehabilitation System for Parkinson's Patients Based on YOLOv5

This project is a YOLOv5-based game assistance program for visual recognition.

---

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [FAQ](#faq)
- [Contact](#contact)

---

## Introduction

This project provides an innovative rehabilitation system for Parkinson's patients. By integrating YOLOv5-based target detection with **Counter-Strike 2 (CS2)** gameplay, it offers a fun and effective way to improve hand control. Key goals include:

1. **Rehabilitation Support**: Enhance hand control and reduce tremors with adjustable smoothing factors.
2. **Engaging Experience**: Use FPS gameplay to increase patient motivation and involvement.
3. **Simplified Gameplay**: Automatic aiming and dynamic adjustments lower the difficulty for patients.

This system merges rehabilitation with entertainment, aiming to improve both health and quality of life.

---

## Features

### Target Detection and Customization
- **Real-time Target Detection**: YOLOv5 identifies targets in a 500x500 pixel screen center.
- **Customizable Smoothing Factor**: Adjust mouse smoothing (recommended range: 0.1-2.0) for different stability levels.

### User Experience Enhancements
- **Dynamic Mouse Adjustment**: Automatically centers the mouse on detected targets.
- **Visual Feedback**: Detection boxes and crosshair (red dot) provide clear visual guidance.
- **FPS Display**: Real-time FPS shown in the program window.

### Performance Optimization
- **Multithreading**: Separates screen capture and target detection to reduce lag.
- **Weapon Adaptability**: Supports FPS weapon modes like sniper and SMG.

### Accessibility Features
- **Hotkey Controls**: Start or stop the program with a single hotkey.
- **Always-on-Top Window**: Program window stays visible during gameplay.
- **Future Compatibility Alerts**: Informs users about potential YOLO/Torch updates.
  
---

## Quick Start

1. Set **CS2** to windowed mode.
2. Run the program:
   ```bash
   python main_program.py
   ```
3. Enter the smoothing factor when prompted (default: 0.5).
4. Enjoy automatic aiming while playing the game!

---

## Installation

Follow these steps to set up the YOLOv5 environment:

1. **Clone the YOLOv5 Repository**:
   ```bash
   git clone https://github.com/ultralytics/yolov5.git
   cd yolov5
   ```

2. **Create a Virtual Environment (Optional)**:
   ```bash
   python -m venv yolov5_env
   source yolov5_env/bin/activate  # Linux/macOS
   yolov5_env\Scripts\activate     # Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**:
   ```bash
   python detect.py --source data/images --weights yolov5s.pt --conf 0.25
   ```
   Ensure the script runs successfully.
---

## Usage

1. **Prepare the Environment**:
   - Install **Python** (recommended: version 3.8 or higher).
   - Set up the YOLOv5 environment.

2. **Configure the Program**:
   - Update file paths in `main_program.py`:
     ```python
     yolov5_code_path = 'D:/code/ai-aim-ver1/yolov5'  # Path to YOLOv5 code, replace it with the path to the yolov5_run folder on your computer
     model_weights_path = 'D:/code/ai-aim-ver1/best.pt'  # Path to the model file, replace it with the path to the 9may_yolov5s_final folder on your computer
     ```

3. **Launch CS2**:
   - Set the game to **windowed mode** in Steam properties:
     ```
     -windowed
     ```

4. **Run the Program**:
   ```bash
   python main_program.py
   ```

5. **Adjust Settings**:
   - Enter the smoothing factor when prompted (recommended: 0.1 to 2.0).

6. **Exit**:
   - Press **"0" key** to stop the program.

---

## Project Structure

### Weights
- **7may-yolov5s**: Initial test.
- **9may_yolov5x**: v5x model test.
- **9may_yolov5s_final**: Final weights.
- **yolov5s**: Pretrained baseline weights.
- **yolov8n**: Early test weights.

### Program
- **main_program**: Core program.

### Documentation
- **README**: Project documentation.
- **info**: Internal reference notes.

### Testing
- **testyolov8**: Early YOLOv8 validation.

### YOLO Environment
- **yolov5**: YOLOv5 training environment.
- **yolov5_run**: YOLOv5 folder for running `main_program`.

### Git information
- **.git**: Don't delete it.
  
---

# Contributing

We welcome contributions! Since this project is shared offline, follow these steps:

1. **Get the Latest Code**:
   - Contact the maintainer for the latest version.

2. **Make Changes**:
   - Clearly comment on changes and their purpose.

3. **Submit Updates**:
   - Share the updated files via email or file-sharing tools.

4. **Guidelines**:
   - Ensure compatibility and proper documentation.
   - Back up original files before changes.

---

## License

This project does not currently include a specific license. Contributions are subject to the maintainer’s discretion.

---

## Acknowledgments

Thanks to the **YOLOv5** team for their excellent work in target detection.

- **YOLOv5 Repository**: [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)

This project builds upon their pretrained models and code to meet unique rehabilitation needs.

---

# FAQ

### Q: Does this work on MacOS?
A: Yes, but ensure all dependencies are installed correctly.

### Q: Can I use this with other FPS games?
A: It is optimized for CS2 but can be adapted with some configuration.

### Q: What if the program crashes?
A: Verify paths and ensure dependencies are installed.

---

## Contact

For questions or issues, contact the project maintainer.
