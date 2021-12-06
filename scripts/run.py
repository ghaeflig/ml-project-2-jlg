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
    pred = model(img)
    pred = pred.squeeze(1)
    pred = torch.sigmoid(pred)
    pred[pred>0.25] = 1
    pred[pred<=0.25] = 0
    return pred

def img_float_to_uint8(img):
    """transform img from float to uint8"""
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg



# Load the test set 
test_dir = "../data/test_set_images/"
test_set = TestDataset(test_dir)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)


# Load checkpoint
root_dir = "../outputs/"  
output_dir =  "output_NE25_BS8_LR1e-05_WD1e-08" #OUTPUT_DIR
checkpoint_path = os.path.join(root_dir, output_dir + '/parameters.pt')
#print(checkpoint_path)
model = UNET().to(DEVICE)
#print("You are on device : {}".format(DEVICE))
load_checkpoint(checkpoint_path, model)
model.eval() #a voir 

# Make and save test predictions
save_path = os.path.join(root_dir, output_dir + '/prediction_masks')

os.mkdir(save_path)
#print(save_path)
it = 1
#for idx, batch_x in test_loader :
for batch_x in test_loader :
	#print('boucle test loader entry')
	batch_x = batch_x.permute(0, 3, 2, 1).float()
	batch_x = batch_x.to(DEVICE)
	pred = get_prediction(batch_x, model)
	for i in range(BATCH_SIZE) : 
		#print('boucle batch entry')
		mask = pred[i].cpu().detach().numpy() #we can not convert cuda tensor into numpy
		mask = img_float_to_uint8(mask)
		mask_path = save_path + "/test_pred_{}.png".format(it)
		#print(mask_path)
		io.imsave(mask_path, mask) 
		print('Prediction mask for test image {} is saved'.format(it))
		it = it + 1

# Make a submission csv file
sub_path = os.path.join(root_dir, output_dir)
submission(save_path, sub_path)



