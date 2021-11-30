import numpy as np
import os
from PIL import Image, ImageOps
from scipy import ndimage, misc
import cv2
import random

class DataAugm():
    """ Get more data by rotating, flipping the images and adding gaussian noise"""
    def __ini__(self, args, image, gt):
        self.image = image
        self.gt = gt

    def rotation(degree, image, gt):
        "Returns the rotated original image and groundtruth"
        img_rot = ndimage.rotate(image, degree, reshape=False, mode='reflect')
        gt_rot = ndimage.rotate(gt, degree, reshape=False, mode='reflect')
        return img_rot, gt_rot

    # flipping
    def flip(image, gt):
        "Returns the flipped original image and groundtruth"
        img_flip = np.flip(image, 1)
        gt_flip = np.flip(gt, 1)
        return img_flip, gt_flip

    # change in colors

    # gaussian noise
    def gaussian_noise(image):
        "Returns the original image and groundtruth with added gaussian noise"
        mean = 0
        var = 5
        sigma = var**0.5
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
    print(ids)
    augm_idx = len(ids) + 1 #because images index strat at 1 (=/ 0 for python arrays)

    images = os.listdir(image_dir) #list all files in that folder
    gts = os.listdir(gt_dir)
    print(f'Loading {len(ids)} samples')
    nb_rot = 0
    print('Performing data augmentation')
    for i in np.arange(len(ids)-1):
        image_path = os.path.join(image_dir, images[i])
        gt_path = os.path.join(gt_dir, images[i])
        image = np.array(Image.open(image_path).convert("RGB"))
        gt = np.array(Image.open(gt_path).convert("L"), dtype=np.float32)

        # for each image, creation of 5 random rotated images
        for i in np.arange(5): # for each image, creation of 5 random rotated images
            deg = random.randint(20, 160)
            rotImage, rotGt = DataAugm.rotation(deg, image, gt)
            rotImage = Image.fromarray(rotImage).convert('RGB')
            rotGt = Image.fromarray(rotGt).convert('RGB')
            rotImage.save(image_dir+"/rotImage_"+"{:03}".format(augm_idx)+".png", "PNG")
            rotGt.save(gt_dir+"/rotImage_"+"{:03}".format(augm_idx)+".png", "PNG")
            augm_idx += 1
            nb_rot +=1

        # for each image, creation of 1 flipped images
        fliImage, fliGt = DataAugm.flip(image, gt)
        fliImage = Image.fromarray(fliImage).convert('RGB')
        fliGt = Image.fromarray(fliGt).convert('RGB')
        rotImage.save(image_dir+"/fliImage_"+"{:03}".format(augm_idx)+".png", "PNG")
        rotGt.save(gt_dir+"/fliImage_"+"{:03}".format(augm_idx)+".png", "PNG")
        augm_idx += 1

        # for each image, creation of 1 image with added noise
        noiImage = DataAugm.gaussian_noise(image)
        noiImage = Image.fromarray(noiImage).convert('RGB')
        noiGt = gt
        rotImage.save(image_dir+"/noiImage_"+"{:03}".format(augm_idx)+".png", "PNG")
        rotGt.save(gt_dir+"/noiImage_"+"{:03}".format(augm_idx)+".png", "PNG")
        augm_idx += 1

    print("{nb} rotated images and groundtruths saved".format(nb=augm_idx-100))
    print("{nb} flipped images and groundtruths saved".format(nb=len(ids)))
    print("{nb} noisy images and groundtruths saved".format(nb=len(ids)))
    print("Data augmentation finished - {nb} added samples".format(nb=augm_idx-100))

if __name__ == "__main__":
    test()
