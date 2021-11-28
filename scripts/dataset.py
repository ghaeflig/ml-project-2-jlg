import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from helpers import value_to_class
from helpers import transform

class ImgDataset(Dataset):
    """ Dataset loader
    train or validation mode 
    """
    def __init__(self, image_dir, gt_dir, mode="train", split_ratio=0.8, transform=None):
        self.mode = mode
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        
        ids = [os.path.splitext(file)[0] for file in os.listdir(image_dir)]
        ids.sort() #shuffle , is in loader???
        n = len(ids)
        if self.mode == "train" :
            self.ids = ids[:int(n * split_ratio)]
        if self.mode == "val": 
            self.ids = ids[int(n * split_ratio):]
            
        self.images = os.listdir(image_dir) #list all files in that folder
        self.gt = os.listdir(gt_dir)
        print(f'Creating {mode} dataset with {len(self.ids)} samples')
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        if self.mode == "test":
            image_path = os.path.join(self.test_dir, self.ids[index])
            image = np.array(Image.open(image_path).convert("RGB"))
        else:
            image_path = os.path.join(self.image_dir, self.images[idx])
            gt_path = os.path.join(self.gt_dir, self.images[idx])
            image = np.array(Image.open(image_path).convert("RGB"))
            gt = np.array(Image.open(gt_path).convert("L"), dtype=np.float32)
            gt = value_to_class(np.mean(gt), 0.25) # for val too?
            if transform is not None:
                image, gt = tranform(image, gt) # returns a vector of values is ok? need to change self.images?      
        # Preprocessing: data augmentation, balancing, cut in patches = transform ?
        
        
             
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
        
               


           

            

            


