import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
from helpers import value_to_class

class ImgDataset(Dataset):
    def __init__(self, image_dir, gt_dir, transform=None):
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.images = os.listdir(image_dir) #list all files in that folder
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        gt_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        gt = np.array(Image.open(gt_path).convert("L"), dtype=np.float32)
        gt = value_to_class(np.mean(gt), 0.25) #########################
        
        if self.transform is not None:
            augmentations = self.transform(image=image, gt=gt)
            image = augmentation["image"]
            gt = augmentation["groundtruth"]
            


