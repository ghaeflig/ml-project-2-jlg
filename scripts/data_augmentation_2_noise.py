import numpy as np
import os
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
from scipy import ndimage, misc
import skimage
import cv2
import random

class DataAugm():
    """ Get more data by rotating, flipping the images and adding gaussian noise"""
    def __ini__(self, args, image, gt):
        self.image = image
        self.gt = gt

    def gaussian_noise(image):
        "Returns the original image and groundtruth with added gaussian noise"
        """img_noisy = np.zeros(image.shape, np.float32)
        for i in np.arange(3):
            img_noisy[:, :, i] = skimage.util.random_noise(image[:, :, i], mode = 'gaussian')
        img_noisy = img_noisy.astype(np.uint8)"""

        mean = 0
        var = 1
        sigma = np.std(image)
        gaussian = np.random.normal(mean, sigma, (image.shape[0], image.shape[1]))
        img_noisy = np.zeros(image.shape, np.float32)

        for i in np.arange(3):
            img_noisy[:, :, i] = image[:, :, i] + gaussian

        cv2.normalize(img_noisy, img_noisy, 0, 255, cv2.NORM_MINMAX, dtype=-1)
        img_noisy = img_noisy.astype(np.uint8)

        return img_noisy

def test():
    root_dir = "../data/training/"
    image_dir = root_dir + "images/"
    gt_dir = root_dir + "groundtruth/"

    ids = [int(file[9:12]) for file in os.listdir(image_dir)]
    ids.sort()
    ids=ids[:100]
    augm_idx = len(ids) + 1 #because images index strat at 1 (=/ 0 for python arrays)

    images = os.listdir(image_dir) #list all files in that folder
    gts = os.listdir(gt_dir)
    print(f'Loading {len(ids)} samples')
    nb_rot = 0
    print('Performing data augmentation')
    for i in np.arange(len(ids)):
        image_path = os.path.join(image_dir, images[i])
        gt_path = os.path.join(gt_dir, images[i])
        image = np.array(Image.open(image_path).convert("RGB"))
        gt = np.array(Image.open(gt_path).convert("L"), dtype=np.float32)

        # for each image, creation of 1 image with added noise
        noiImage = DataAugm.gaussian_noise(image)
        noiImage = Image.fromarray(noiImage).convert('RGB')
        noiGt = Image.fromarray(gt).convert('RGB')
        noiImage.save(image_dir+"/noiImage_"+"{:03}".format(augm_idx)+".png", "PNG")
        noiGt.save(gt_dir+"/noiImage_"+"{:03}".format(augm_idx)+".png", "PNG")
        augm_idx += 1


    print("{nb} noisy images and groundtruths saved".format(nb=len(ids)))

if __name__ == "__main__":
    test()
