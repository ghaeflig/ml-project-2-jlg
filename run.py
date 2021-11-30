import numpy as np
import os, sys
from torch.utils import data
from dataset import ImgDataset
from dataset import TestDataset
from helpers import *
from model import UNET
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CHANNELS = 3  # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 20
VALIDATION_SIZE = 5  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 16  # 64
NUM_EPOCHS = 2
RESTORE_MODEL = False  # If True, restore existing model instead of training a new one
RECORDING_STEP = 0
PATCH_SIZE = 16
#CRITERION =

# Loaded a set of images
root_dir = "training/"
image_dir = root_dir + "images/"
gt_dir = root_dir + "groundtruth/"
test_dir = "test_set_images/"

train_set = ImgDataset(image_dir, gt_dir, split_ratio=0.8, mode="train")
val_set = ImgDataset(image_dir, gt_dir, split_ratio=0.8, mode="val")
test_set = TestDataset(test_dir)

# Constructing the dataset from our map-style dataset
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# Essaia
def make_img_overlay(img, predicted_img): #superposition
    w = img.shape[0]
    h = img.shape[1]
    color_mask = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    color_mask[:, :, 0] = predicted_img*PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def check_accuracy(pred, y):
    pred = (pred > 0.5).float()
    num_correct = 0
    num_pixels = 0
    num_correct += (pred == y).sum()
    num_pixels += torch.numel(pred)
    return num_correct/num_pixels*100

def train_func(train_loader, model, epoch, criterion, optimizer, scaler=None, writer=None, device=DEVICE) : #rajouter scaler if cuda
    model.train()
    train_loss = 0
    accuracies = 0
    #it = 0
    #batch_x = train images and batch_y = train masks
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.permute(0, 3, 2, 1).float()
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        # Evaluate the network (forward pass)
        pred = model(batch_x).squeeze(1)

        # Compute the loss and the gradient
        loss = criterion(pred, batch_y)
        train_loss += float(loss.item())
        acc = check_accuracy(torch.sigmoid(pred), batch_y)
        accuracies += acc   #jsp s'il faut sigmoid ici ??
        optimizer.zero_grad()
        loss.backward()

        # Scaler prevents underflow ; gradient values have a larger magnitude, so they don’t flush to zero
        if scaler is not None :
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Update the parameters of the model with a gradient step
        optimizer.step()

        #if writer is not None:
            #writer.add_scalar("Train loss / batch for epoch {}".format(epoch), loss.item(), it)
        #it = it + 1

    train_loss = train_loss / len(train_loader)
    accuracy = accuracies/len(train_loader)
    if writer is not None :
        writer.add_scalar('Train loss / epoch', train_loss, epoch)
        writer.add_scalar('Train accuracy / epoch', accuracy, epoch)
    return train_loss, accuracy

def train_val(val_loader, model, epoch, writer=None, device = DEVICE) :
    model.eval()
    accuracies = 0
    with torch.no_grad() :
        for batch_x, batch_y in val_loader :
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred = torch.sigmoid(model(batch_x))
            accuracies += check_accuracy(pred, batch_y)

    accuracy_avg = accuracies/len(val_loader)
    if writer is not None :
        writer.add_scalar("Validation accuracy / epoch", accuracy_avg, epoch)
    return accuracy_avg

def training(train_loader, val_loader, print_err=True) :
    print("\nStart training...")
    if not torch.cuda.is_available() :
        print("\nThings will go much quicker if you enable a GPU in Colab under 'Runtime / Change Runtime Type'")
    else :
        print("\nYou are running the training of the data on a GPU")

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #scheduler reduces learning rate when a metric has stopped improving
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7, verbose=True)
    writer = SummaryWriter() # folder location: runs/May04_22-14-54_s-MacBook-Pro.comment/ comment=''
    scaler = None
    if torch.cuda.is_available():
        scaler_ = torch.cuda.amp.GradScaler()
        print("\nAs you are on a GPU, a scaler is added to the training in order to prevent underflow")

    for epoch in range(NUM_EPOCHS):
        max_accuracy = 0
        train_loss, accuracy_train = train_func(train_loader, model, epoch, criterion, optimizer, scaler, writer)
        accuracy = train_val(val_loader, model, epoch, writer)

        if (print_err == True) :
            print("\nEpoch {} | Train loss: {:.5f} and train accuracy: {:.5f}".format(epoch, train_loss, accuracy_train))
            print("\nEpoch {} | Validation accuracy: {:.5f}".format(epoch, accuracy))
        #scheduler.step()
        if accuracy > max_accuracy :
            max_accuracy = accuracy.copy()
            max_accuracy_epoch = epoch.copy()
            torch.save(model.state_dict(),  OUTPUT_DIR + '/parameters.pt') #.pt or .plk

    print("\nThe maximum accuracy over all epochs is {} at epoch {}.".format(max_accuracy, epoch))
    print("\nThe best model over all epochs is saved into folder name {} with name parameters.pt".format(OUTPUT_DIR))


training(train_loader, val_loader)
