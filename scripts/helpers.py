import torch
import torchvision
from dataset import ImgDataset
from torch.utils.data import Dataloader

def value_to_class(v, foreground_threshold): #################################
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0
    
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])    
    
def get_loaders(train_dir, train_gtdir, val_dir, val_gtdir, 
    batch_size, train_transform, val_transform, num_workers=4, pin_memory=True):
    train_ds = ImgDataset(
        image_dir=train_dir, gt_dir=train_gtdir, transform=train_transform)
    train_loader = DataLoader( 
        train_ds, batch_size=batch_size, num_workers=num_workers,
        pin_memory=pin_memory, shuffle=True)
    val_ds = ImgDataset(
        image_dir=val_dir, gt_dir=val_gtdir, transform=val_transform)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, num_workers=num_workers,
        pin_memory=pin_memory, shuffle=False)
    
    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float
            num_correct += (preds == y).sum
            num_pixels += torch.numel(preds)
            
     print(
         f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100}:.2f")
    model.train()
    
def save_predictions_as_imgs():
    
    


    

    