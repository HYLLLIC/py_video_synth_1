import pygame
import numpy as np
import cv2
import os
from datetime import datetime

# Configurable parameters
WINDOW_SIZE = (800, 800)
GRID_SIZE = (64, 64)  # The "pixel" resolution of the pattern
PIXEL_SIZE = (WINDOW_SIZE[0] // GRID_SIZE[0], WINDOW_SIZE[1] // GRID_SIZE[1])
FPS = 30
