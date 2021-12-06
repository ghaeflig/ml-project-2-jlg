import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
#from helpers import value_to_class
#from data_augmentation import transform

class ImgDataset(Dataset):
    """ Dataset loader
    train or validation mode
    """
    def __init__(self, image_dir, gt_dir, mode="train", split_ratio=0.8, transform=None):
        self.mode = mode
        self.image_dir = image_dir
        self.gt_dir = gt_dir

        #ids = [os.path.splitext(file)[0] for file in os.listdir(image_dir)]
        ids = [int(file[9:12]) for file in os.listdir(image_dir)]
        ids.sort() #shuffle , is in loader???
        n = len(ids)
        if self.mode == "train" :
            self.ids = ids[0:int(n * split_ratio)]
            #print(f'ids: {self.ids}')
        if self.mode == "val":
            self.ids = ids[int(n * split_ratio):n]
            #print(f'ids: {self.ids}')
            #print(f'TEST index 0 val: {self.ids[0]-1}')

        self.images = os.listdir(image_dir) #list all files in that folder
        self.gt = os.listdir(gt_dir)
        print(f'Creating {mode} dataset with {len(self.ids)} samples')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        #print(f'index: {index}')
        idx = self.ids[index]-1 # bc index [0,99] and ids [1,100] and images[idx] needs [0,99]
        #print(f'idx: {idx}')
        image_path = os.path.join(self.image_dir, self.images[idx])
        gt_path = os.path.join(self.gt_dir, self.images[idx])
        image = np.array(Image.open(image_path).convert("RGB"))
        gt = np.array(Image.open(gt_path).convert("L"), dtype=np.float32)
        #gt = np.expand_dims(gt, 0)
        gt[gt>0.25] = 1
        gt[gt<=0.25] = 0
        #if transform is not None:
            #image, gt = transform(image, gt) # returns a vector of values is ok? need to change self.images?
            # Preprocessing: data augmentation, balancing, cut in patches = transform ?
        return image, gt

class TestDataset(Dataset):
    """ Test dataset loader
    """
    def __init__(self, test_dir):
        self.dir = test_dir
        self.images = os.listdir(test_dir)
        print(f'Creating test dataset')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.dir, self.ids[index])
        image = np.array(Image.open(image_path).convert("RGB"))
        return image
