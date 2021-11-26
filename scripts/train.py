import torch
#import albumentations as A
#from albumentations.pytorch import ToTensorV2
#from tqdm import tqdm
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
#rom model import UNET
from helpers import *

# Hyperparameters etc.
"""LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/""""

""""LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CHANNELS = 3  # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 20
VALIDATION_SIZE = 5  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 16  # 64
NUM_EPOCHS = 100
RESTORE_MODEL = False  # If True, restore existing model instead of training a new one
RECORDING_STEP = 0
#PATCH_SIZE = ?
#CRITERION = ?"""

SEED = 66478
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 100
BATCH_SIZE = 16 
LEARNING_RATE = 1e-4

torch.manual_seed(SEED)

""""train_set = ImgDataset(args, mode='train')
valid_set = ImdDataset(args, mode='valid')
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
valid_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=BATCH, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)"""

"""Or :
train_loader, valid_loader = get_loaders(train_dir, train_gtdir, val_dir, val_gtdir, 
    batch_size, train_transform, val_transform, num_workers=4, pin_memory=True)"""

def accuracy(pred) :
    pred = (pred > 0.5).float()
    num_correct = 0
    num_pixels = 0
    num_correct += (pred == y).sum()
    num_pixels += torch.numel(pred)
    return num_correct/num_pixels*100


def train_func(train_loader, model, epoch, criterion, optimizer, device=DEVICE) :

        
        model.train()
        train_loss = 0
        accuracies = 0
        # batch_x = train images and batch_y = train masks
        for i, (batch_x, batch_y) in ennumrate(train_loader) :  #for batch_x, batch_y in train_loader :
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Evaluate the network (forward pass)
            pred = model(batch_x)
      
            # Compute the loss and the gradient
            loss = criterion(pred, batch_y)
            train_loss += float(loss.item())
            accuracies += accuracy(torch.sigmoid(pred))
            optimizer.zero_grad()  
            loss.backward()
            
            # Update the parameters of the model with a gradient step
            optimizer.step()
            
        train_loss = train_loss / len(train_loader)
        print("Epoch {} | Train loss: {:.5f}".format(epoch, train_loss))
        print("Epoch {} | Train accuracy: {:.5f}".format(epoch, accuracies/len(train_loader))) #jsp s'il faut sigmoid ici ??
        


def train_val(val_loader, epoch, device = DEVICE) :
    model.eval()
    accuracies = 0
    with torch.no_grad() :
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            pred = torch.sigmoid(model(batch_x))
            accuracies += accuracy(pred)
        print("Epoch {} | Validation accuracy: {:.5f}".format(epoch, accuracies/len(val_loader)))
        
        
def main() :
    #train_loader, val_loader = get_loader(...)
    #model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(NUM_EPOCHS):
        train_func(train_loader, model, epoch, criterion, optimizer)
        train_val(val_loader, epoch)

        
        
