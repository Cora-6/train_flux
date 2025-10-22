# check_stitch.py
import os
import sys
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from cnet_dataset_flux import HSIControlNetDataset   # 你上面给的完整 Dataset
from time import time

# 1. 构造一个“迷你数据集”：只要 4 个视角即可
mini_root = "../data/train/"          # 改成你的
ds = HSIControlNetDataset(mini_root, image_size=512)
print(len(ds))