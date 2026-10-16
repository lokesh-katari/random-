# 3D Spatial Design Canvas (Excalidraw Aesthetic + Hand Tracking)

A GPU-accelerated 3D spatial canvas built with **Ursina Engine / Panda3D**, featuring real-time hand gesture tracking via **OpenCV** and **MediaPipe HandLandmarker**, styled with a hand-drawn Excalidraw wireframe aesthetic.

## Features
- **Excalidraw Aesthetic**: Double-stroke line wireframe meshes with micro-jitter/roughness simulating hand-drawn architectural assets (boxes, cylinders, clouds, connectors).
- **Computer Vision & Hand Tracking**: Multi-threaded MediaPipe hand tracking using OpenCV video capture.
- **Gesture Controls**:
  - **Pinch to Grab**: Index finger + thumb pinch to select and translate 3D objects in space.
  - **Two-Hand Scale & Rotate**: Dual-hand pinch on the same object to rotate and zoom.
- **Anti-Jitter Smoothing**: Lerp-smoothed cursor and object physics for responsive manipulation.

## Installation & Running

```bash
# Clone the repository
git clone https://github.com/lokesh-katari/spatial-canvas.git
cd spatial-canvas

# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the spatial canvas
python3 spatial_canvas.py
```
