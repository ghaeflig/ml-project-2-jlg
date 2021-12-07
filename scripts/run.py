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

IMG_PLOTS = False #save des images avec overlay ou mask

def get_prediction(img, model) :
    pred = model(img)
    pred = pred.squeeze(1)
    pred = torch.sigmoid(pred)
    pred[pred>0.5] = 1
    pred[pred<=0.5] = 0
    return pred

def img_float_to_uint8(img) : #a mettre dans helper
    """transform img from float to uint8"""
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


print('\nStart running...')
if not torch.cuda.is_available() :
	print("\nThings will go much quicker if you enable a GPU in Colab under 'Runtime / Change Runtime Type'")
else :
	print("\nYou are running the prediction of the test data on a GPU")

# Load the test set 
print('\nTest data loading...')
test_dir = "../data/test_set_images/"
test_set = TestDataset(test_dir)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)


# Load checkpoint
root_dir = "../outputs"  
#output_dir =  "" #"output_NE25_BS8_LR1e-05_WD1e-08" name of the folder 
checkpoint_path = os.path.join(root_dir, 'parameters.pt')
print('checkpoint path {}'.format(checkpoint_path))
model = UNET().to(DEVICE)
load_checkpoint(checkpoint_path, model)

# Model evaluation
model.eval() 

# Make and save test binary prediction images
save_path = os.path.join(root_dir, 'binary_masks')
print('save path {}'.format(save_path))
os.mkdir(save_path)
it = 1
for batch_x in test_loader :
	batch_x = batch_x.permute(0, 3, 2, 1).float()
	batch_x = batch_x.to(DEVICE)
	pred = model(batch_x)
	pred = pred.squeeze(1)
	pred = torch.sigmoid(pred)
	for i in range(BATCH_SIZE) : 
		mask = pred[i].cpu().detach().numpy() #we can not convert cuda tensor into numpy
		#print(mask.type())
		mask[mask>0.5] = 1
		mask[mask<=0.5] = 0
		#mask = img_float_to_uint8(mask)
		mask_path = os.path.join(save_path, '%.3d' % it + '.png')
		print(mask_path)
		io.imsave(mask_path, mask) 
		print('Prediction binary mask for test image {} is saved'.format(it))
		it = it + 1
print('\nYou can find all the test binary masks by following the path : {}'.format(save_path))

# Make a submission csv file
print('\nTransformation of the binary masks into a submission csv file...')
submission(save_path, root_dir)
print('\nYou can find the submission.csv file by following the path : {}'.format(root_dir))

#if IMG_PLOTS :




