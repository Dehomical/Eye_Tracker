# Real-time Eye Tracking System

A real-time eye tracking system based on YOLOv8 and MediaPipe, implementing face detection, eye localization, blink detection, and gaze direction estimation.

## Features

- Face Detection: YOLOv8 for fast face region localization
- Eye Localization: MediaPipe 468 3D facial landmarks for precise eye positioning
- Blink Detection: Real-time blink detection based on eyelid vertex distance
- Pupil Tracking: CLAHE enhancement with contour detection for pupil center localization
- Gaze Estimation: 8-direction gaze estimation including combinations
- GPU Acceleration: PyTorch CUDA acceleration for 2x FPS improvement
- Real-time Statistics: FPS, detection rate, and blink count display

## Tech Stack

| Technology | Purpose |
|------------|---------|
| YOLOv8 | Face detection |
| MediaPipe | 468 3D facial landmarks localization |
| OpenCV | Image processing and UI drawing |
| PyTorch | GPU acceleration |
| NumPy | Numerical computation |

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/eye-tracking-system.git
cd eye-tracking-system
