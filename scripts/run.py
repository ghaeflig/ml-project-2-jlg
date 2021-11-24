import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
from dataset import ImgDataset 
from helpers import *


# Loaded a set of images
root_dir = "training/"

image_dir = root_dir + "images/"
files = os.listdir(image_dir)
print("Loading images")
imgs = [load_image(image_dir + files[i]) for i in range(len(files))]

gt_dir = root_dir + "groundtruth/"
print("Loading groundtruth images")
gt_imgs = [load_image(gt_dir + files[i]) for i in range(len(files))]

