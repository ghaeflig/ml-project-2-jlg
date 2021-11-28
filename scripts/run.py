import numpy as np
import os
from PIL import Image
from torch.utils import data
from dataset import ImgDataset
from dataset import TestDataset
from helpers import *

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CHANNELS = 3  # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 20
VALIDATION_SIZE = 5  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 16  # 64
NUM_EPOCHS = 100
RESTORE_MODEL = False  # If True, restore existing model instead of training a new one
RECORDING_STEP = 0
PATCH_SIZE = 16
#CRITERION = 

# Loaded a set of images
root_dir = "training/"

image_dir = root_dir + "images/"
gt_dir = root_dir + "groundtruth/"
test_dir = "test_set_images/"

train_set = ImgDataset(image_dir, gt_dir, split_ratio=0.8, mode="train")
val_set = ImgDataset(image_dir, gt_dir, split_ratio=0.8, mode="val")
test_set = TestDataset(test_dir)

# Constructing the dataset from our map-style dataset
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)



