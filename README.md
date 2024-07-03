# Canny Edge Detector

## Overview

This Python script implements a basic Canny Edge Detector for image processing. The Canny Edge Detector is a multi-step process that involves smoothing an image, computing gradients, non-maximum suppression, and thresholding to detect edges in an image.

## Features

- **Image Loading**: Load and preprocess an image.
- **Gaussian Filtering**: Apply Gaussian smoothing to reduce noise.
- **Gradient Calculation**: Compute image gradients using Sobel filters.
- **Non-Maximum Suppression**: Thin out edges by suppressing non-maximum pixels.
- **Thresholding**: Apply hysteresis thresholding to detect strong and weak edges.
- **Visualization**: Display the results using Matplotlib.

## Requirements

- Python 3.x
- OpenCV: Install via `pip install opencv-python`
- NumPy: Install via `pip install numpy`
- Matplotlib: Install via `pip install matplotlib`

