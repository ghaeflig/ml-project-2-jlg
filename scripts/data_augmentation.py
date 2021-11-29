import numpy as np
import os
from PIL import Image, ImageOps
from scipy import ndimage, misc

class DataAugm():
    """ Get more data by rotating and flipping the images """
    def __ini__(self, args, image, gt):

    def rotation(degree):
        # rotation
        # import the Python Image processing Library
        # Create an Image object from an Image
        # Find a solution to open images here or pass adress to functions
        colorImage  = Image.open("training/images/satImage_015.png")
        # Rotate it by 45 degrees
        img_45 = ndimage.rotate(colorImage, degree, reshape=False, mode='reflect')
        gt_45 = ndimage.rotate(gt, degree, reshape=False, mode='reflect')

    # flipping
    def flip():
        im_flip = ImageOps.flip(image)
        gt_flip = ImageOps.flip(image)

    # change in colors

    # noise injection?



for i in range()
    im1 = im1.save("satImage_i_rotated_deg")
