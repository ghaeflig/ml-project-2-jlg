import torch
import torchvision
import os
import numpy as np
from train import DEVICE


############################ PREPROCESSING ###########################################
def value_to_class(v, foreground_threshold):
    """ Get a one-hot vector for the classes """
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0

def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

def transform(imgs, gt_imgs):
    # Compute features for each image patch
    # Extract patches from input images
    patch_size = 16 # each patch is 16*16 pixels

    img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(n)]
    gt_patches = [img_crop(gt_imgs[i], patch_size, patch_size) for i in range(n)]

    # Linearize list of patches
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])

    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

    X = np.asarray([extract_features_2d(img_patches[i]) for i in range(len(img_patches))])
    Y = np.asarray([value_to_class(np.mean(gt_patches[i]), foreground_threshold) for i in range(len(gt_patches))])

    # Print feature statistics
    print('Computed ' + str(X.shape[0]) + ' features')
    print('Feature dimension = ' + str(X.shape[1]))
    print('Number of classes = ' + str(np.max(Y)))  #TODO: fix, length(unique(Y))

    Y0 = [i for i, j in enumerate(Y) if j == 0]
    Y1 = [i for i, j in enumerate(Y) if j == 1]
    print('Class 0: ' + str(len(Y0)) + ' samples')
    print('Class 1: ' + str(len(Y1)) + ' samples')

    # Display a patch that belongs to the foreground class i.e with ones
    plt.imshow(gt_patches[Y1[2]], cmap='Greys_r')

    # Display a patch that belongs to the background class i.e mostly zeros
    #plt.imshow(gt_patches[Y0[2]], cmap='Greys_r')

    # Balancing and extract features?
    return img_patches, gt_patches

############################# CHECKPOINTS ###########################################
def save_checkpoint(save_path, epoch, model, optimizer, scheduler, scaler) :
    print("=> Saving checkpoint")
    saved_scheduler = None
    saved_scaler = None
    if scheduler is not None:
        saved_scheduler = scheduler.state_dict()
    if scaler is not None :
        saved_scaler = scaler.sate_dict()
    checkpoint = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': saved_scheduler, 'scaler': saved_scaler}
    torch.save(checkpoint, save_path + '/parameters.pt')
    
    
def load_checkpoint(checkpoint_path, model, optimizer = None, scheduler = None, scaler = None):
    if not os.path.exists(checkpoint_path):
        print("The requested file ({}) does not exist ; the checkpoint can not be loaded".format(checkpoint_path))
    else :
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        epoch = checkpoint['epoch']
        print("=> Loading checkpoint from a trained model at epoch {}".format(epoch))
        model.load_state_dict(checkpoint['model']) 
        if optimizer is not None :
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None :
            scheduler.load_state_dict(checkpoint['scheduler'])
        if scaler is not None :
            scaler.load_state_dict(checkpoint['scaler'])


############################ MODEL PERFORMANCE EVALUATION ###################################
def save_predictions_as_imgs(s):
    s=3

def make_img_overlay(img, predicted_img): 
    w = img.shape[0]
    h = img.shape[1]
    color_mask = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    color_mask[:, :, 0] = predicted_img*PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img
