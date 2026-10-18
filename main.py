from pathlib import Path
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = Path(__file__).resolve().parent / "SingFlow-R.onnx"


MEAN = np.array(
    [123.675, 116.28, 103.53], dtype=np.float32
).reshape(1, 1, 3)

STD = np.array(
    [58.395, 57.12, 57.375], dtype=np.float32
).reshape(1, 1, 3)

NUM_FRAMES = 32
FRAME_INTERVAL = 2