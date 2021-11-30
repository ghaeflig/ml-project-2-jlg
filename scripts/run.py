import numpy as np
import os, sys
from torch.utils import data
from dataset import TestDataset
from mask_to_submission import *
from submission_to_mask import *



# Loaded a set of images

test_dir = "../data/test_set_images/"
test_set = TestDataset(test_dir)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

