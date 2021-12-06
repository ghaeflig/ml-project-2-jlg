import numpy as np
import os, sys
import torch
import skimage.io as io
from torch.utils import data
from dataset import TestDataset
from helpers import load_checkpoint
from model import UNET
from mask_to_submission import *
from train import DEVICE, BATCH_SIZE, OUTPUT_DIR
#from submission_to_mask import *

def get_prediction(img, model) :
    pred = model(batch_x)
    pred = pred.squeeze(1)
    pred = torch.sigmoid(pred)
    pred[pred>0.25] = 1
    pred[pred<=0.25] = 0
    return pred


# Load the test set 
test_dir = "../data/test_set_images/"
test_set = TestDataset(test_dir)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# Load checkpoint
root_dir = "../outputs/"  
output_dir =  "output_NE25_BS8_LR1e-05_WD1e-08" #OUTPUT_DIR
checkpoint_path = os.path.join(root_dir, output_dir + '/parameters.pt')
print(checkpoint_path)
model = UNET().to(DEVICE)
print("You are on device : {}".format(DEVICE))
load_checkpoint(checkpoint_path, model)
model.eval() #a voir 

# Make and save test predictions
save_path = os.path.join(root_dir, output_dir + '/prediction_masks')
print(save_path)
it = 1
#for idx, batch_x in test_loader :
for batch_x in test_loader :
	batch_x = batch_x.permute(0, 3, 2, 1).float()
	batch_x = batch_x.to(DEVICE)
	pred = get_prediction(batch_x, model)
	io.imsave(os.path.join(save_path, "/test_pred_{}.png".format(it), pred)) #changer it en idx si changement dans test loader
	it = it + 1

# Make a submission csv file
sub_path = os.path.join(root_dir, output_dir)
submission(save_path, sub_path)



