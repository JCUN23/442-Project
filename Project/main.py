import os
import json
import torch
import torchvision
import torchvision.transforms as T
import random
import numpy as np
from object_detection import classify


files = ['images/scene.png']
for f in files:
    label = classify(f)
    print(label)
