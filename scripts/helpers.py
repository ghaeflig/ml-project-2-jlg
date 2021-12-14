import torch
import torchvision
import os
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from train import DEVICE, BATCH_SIZE
from PIL import Image

############################ PREPROCESSING ###########################################
def value_to_class(v, foreground_threshold):
    """ Get a one-hot vector for the classes """
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0

############################ MODEL EVALUATION PERFORMANCE ############################
class IoULoss(torch.nn.Module):
    """Jaccard loss based on https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch"""
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth)/(union + smooth)

        return 1 - IoU

def check_accuracy(pred, y, batch_size=BATCH_SIZE):
    pred = (pred > 0.5).float()
    patch_pred = [img_crop(pred[i].cpu().detach().numpy(), 16, 16) for i in range(batch_size)]
    patch_y = [img_crop(y[i].cpu().detach().numpy(), 16, 16) for i in range(batch_size)]
    return accuracy_score(np.array(patch_y).ravel(), np.array(patch_pred).ravel())

def check_f1(pred, y, batch_size=BATCH_SIZE):
    pred = (pred > 0.5).float()
    patch_pred = [img_crop(pred[i].cpu().detach().numpy(), 16, 16) for i in range(batch_size)]
    patch_y = [img_crop(y[i].cpu().detach().numpy(), 16, 16) for i in range(batch_size)]
    return f1_score(np.array(patch_y).ravel(), np.array(patch_pred).ravel())


############################# CHECKPOINTS ###########################################
def save_checkpoint(save_path, epoch, model, optimizer, scheduler) :
    print("=> Saving checkpoint")
    saved_scheduler = None
    if scheduler is not None:
        saved_scheduler = scheduler.state_dict()
    checkpoint = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': saved_scheduler}
    torch.save(checkpoint, save_path + '/parameters.pt')


def load_checkpoint(checkpoint_path, model, optimizer = None, scheduler = None):
    if not os.path.exists(checkpoint_path):
        print("The requested file ({}) does not exist ; the checkpoint can not be loaded".format(checkpoint_path))
    else :
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        epoch = checkpoint['epoch']
        print("=> Loading checkpoint from a trained model at the best epoch {}".format(epoch))
        model.load_state_dict(checkpoint['model'])
        if optimizer is not None :
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None :
            scheduler.load_state_dict(checkpoint['scheduler'])


############################ PLOTS ###################################
def make_img_overlay(img, predicted_img):
    img = img.permute(1, 2, 0)
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = predicted_img*255

    img8 = img_float_to_uint8(img.detach().numpy())
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def concatenate_images(img, gt_img):
    """ Concatenate an image and its groundtruth or prediction"""
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 4), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        gt_img_3c[:,:,3] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def img_float_to_uint8(img) :
    """transform img from float to uint8"""
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg
