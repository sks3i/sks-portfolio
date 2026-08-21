+++
date = '2026-08-20T23:45:00-07:00'
draft = false
title = 'From 2.5D to 3D: Unprojecting RGB-D Sensor Data for Spatial Memory (Part 1)'
tags = ["computer-vision", "3d-reconstruction", "spatial-computing", "robotics"]
description = "Lifting 2.5D pixels into true 3D space: how to vectorize pinhole camera equations to unproject raw RGB-D data."
+++

In my last post evaluating [SigLIP 2]({{< ref "/posts/siglip2-indoor-place-recognition.md" >}}), the takeaway was pretty clear: pure 2D appearance models fail hard when you start rearranging a room. Global VLMs over-index on object semantics (like matching a red scoop) instead of understanding the actual 3D geometry of the place itself.

If we want an agent to actually understand spatial relationships—or if we want to synthetically edit a scene—we need to lift our data out of the 2.5D image plane and into a true 3D coordinate space. We need continuous geometry.

This is the first of a few posts where I'm building out a differentiable 3D spatial memory pipeline. But before we can manipulate a 3D environment, we have to build the bridge from the hardware. 

### The Pinhole Math 

Standard depth sensors give you two arrays: RGB color, and a registered depth map (usually 16-bit integers representing millimeters). To lift these 2D pixels $(u, v)$ into 3D metric coordinates $(X, Y, Z)$, we use the linear pinhole camera model based on the sensor's intrinsics (focal lengths $f_x, f_y$ and optical center $c_x, c_y$).

$$Z = \text{Depth map value}$$
$$X = \frac{(u - c_x) \cdot Z}{f_x}$$
$$Y = \frac{(v - c_y) \cdot Z}{f_y}$$

### Production-Ready Unprojection 

The NumPy implementation below handles the math instantly, but more importantly, it handles the annoying realities of actual hardware:
1. **Depth Thresholding:** It explicitly filters out `0` values and sets a `max_depth_mm` boundary to drop the noisy, low-confidence points you get from real sensors.
2. **Type Casting:** It safely scales `uint16` millimeter data into the `float32` meters required for standard 3D coordinate spaces.
3. **Color Alignment:** It handles the classic OpenCV BGR-to-RGB flip in a single vectorized slice.

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

DEPTH_SCALE_MM_TO_M = 0.001

def _backproject(
    depth: np.ndarray,
    color_bgr: np.ndarray,
    intrinsics: CameraIntrinsics,
    max_depth_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift a depth map to camera-space 3-D points with per-point RGB colours.

    Args:
        depth:        uint16 (H, W) depth in millimetres.
        color_bgr:    uint8 (H, W, 3) BGR colour image.
        intrinsics:   Pinhole camera parameters.
        max_depth_mm: Points beyond this distance are excluded.

    Returns:
        xyz: float32 (N, 3) array of 3-D points in metres.
        rgb: uint8  (N, 3) array of RGB colours for each point.
    """
    h, w = depth.shape
    u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))

    valid = (depth > 0) & (depth <= max_depth_mm)

    z_mm = depth[valid].astype(np.float32)
    u     = u_coords[valid].astype(np.float32)
    v     = v_coords[valid].astype(np.float32)

    z = z_mm * DEPTH_SCALE_MM_TO_M
    x = (u - intrinsics.cx) * z / intrinsics.fx
    y = (v - intrinsics.cy) * z / intrinsics.fy

    xyz = np.stack([x, y, z], axis=1)                     # (N, 3) float32
    rgb = color_bgr[valid][:, ::-1].astype(np.uint8)      # BGR → RGB, (N, 3)
    return xyz, rgb
```

Running a raw sensor frame through this gives us a clean, metric 3D point cloud of the environment. We've successfully lifted 2.D to 3D using the camera intrinsics. But a raw point cloud is still just a bunch of discrete floating dots. If you try to synthetically move or remove an object from it, you just get a gaping hole in the geometry.

In Part 2, I’ll walk through how to take these exact xyz and rgb arrays, feed them into the Inria PyTorch rasterizer, and initialize the 3D Gaussians. That’s where things get interesting, and we finally move from static dots to a continuous, differentiable world.