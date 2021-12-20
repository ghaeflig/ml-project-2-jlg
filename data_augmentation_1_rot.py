import numpy as np
import os
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
from scipy import ndimage, misc
import skimage
import cv2
import random

class DataAugm():
    """ Get more data by rotating, flipping the images aadding gaussian noise and color augmentation"""
    def __ini__(self, args, image, gt):
        self.image = image
        self.gt = gt

    def rotation(degree, image, gt):
        "Returns the rotated original image and groundtruth. degree should be in degrees"
        img_rot = ndimage.rotate(image, degree, reshape=False, mode='reflect')
        gt_rot = ndimage.rotate(gt, degree, reshape=False, mode='reflect')
        return img_rot, gt_rot

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
    print('Performing data augmentation')
    for i in np.arange(len(ids)-1):
        image_path = os.path.join(image_dir, images[i])
        gt_path = os.path.join(gt_dir, images[i])
        image = np.array(Image.open(image_path).convert("RGB"))
        gt = np.array(Image.open(gt_path).convert("L"), dtype=np.float32)

        deg = np.random.rand()*280 + 40 #random deg between 40 and 320 degrees
        rotImage, rotGt = DataAugm.rotation(deg, image, gt)
        rotImage = Image.fromarray(rotImage).convert('RGB')
        rotGt = Image.fromarray(rotGt).convert('RGB')
        rotImage.save(image_dir+"/rotImage_"+"{:03}".format(augm_idx)+".png", "PNG")
        rotGt.save(gt_dir+"/rotImage_"+"{:03}".format(augm_idx)+".png", "PNG")
        augm_idx += 1

    print("{nb} rotated images and groundtruths saved".format(nb=augm_idx-100))
    print("Data augmentation finished - {nb} added samples".format(nb=augm_idx-100))

if __name__ == "__main__":
    test()
