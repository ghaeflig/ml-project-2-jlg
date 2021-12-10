import numpy as np
import matplotlib.pyplot as plt
import os, sys
import torch.optim as optim
import copy
from dataset import ImgDataset 
from helpers import *
from model import UNET
from torch.utils.tensorboard import SummaryWriter 

SEED = 66478
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 10 
BATCH_SIZE = 2 
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0
OUTPUT_DIR = '../outputs'

# Set the seed
torch.manual_seed(SEED) 


def train_func(train_loader, model, epoch, criterion, optimizer, device=DEVICE) :
    """ Training loop on one epoch for a neurons network model (in our case UNET). The function returns the mean
    training loss error, the training mean accuracy and the training mean f1 score over all batches """
    model.train()
    train_loss = 0
    accuracies = 0
    f1_sum = 0
    it = 0
    for batch_x, batch_y in train_loader: #batch_x = train images and batch_y = train masks (groundtruth)
        print("it num for train func : {} / {}.".format(it, len(train_loader)-1))
        batch_x = batch_x.permute(0, 3, 1, 2).float() # from [8,400,400,3] to [8,3,400,400] for suit the size in the model
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Evaluate the network (forward pass)
        pred = model(batch_x)
        pred = pred.squeeze(1) #[8,1,400,400] to [8,400,400] to have the same size as batch_y
        pred = torch.sigmoid(pred) #if binary cross entropy with logits loss, do sigmoid after the loss calulation
        
        # Compute the loss, the errors and the gradient
        loss = criterion(pred, batch_y) 
        train_loss += float(loss.item())
        accuracies += check_accuracy(pred, batch_y)
        f1_sum += check_f1(pred, batch_y)
        optimizer.zero_grad()

        # Backward  pass
        loss.backward()

        # Update the parameters of the model with a gradient step
        optimizer.step()

        # Adding an iteration
        it = it + 1

    # Compute the mean errors along the batch
    train_loss = train_loss / len(train_loader)
    accuracy_mean = accuracies / len(train_loader)
    f1_mean = f1_sum / len(train_loader)

    return train_loss, accuracy_mean, f1_mean


def train_val(val_loader, model, epoch, device = DEVICE) :
    """ Validation loop on one epoch for a neurons network model (in our case UNET). 
    The function returns validation mean accuracy and the  validation mean f1 score 
    over all batches """
    model.eval()
    accuracies = 0
    f1_sum = 0
    it = 0
    with torch.no_grad() :
        for batch_x, batch_y in val_loader :
            print("it num for train val func : {} / {}.".format(it, len(val_loader)-1))
            batch_x = batch_x.permute(0, 3, 1, 2).float()
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred = model(batch_x)
            pred = pred.squeeze(1)
            pred = torch.sigmoid(pred)
            accuracies += check_accuracy(pred, batch_y)
            f1_sum += check_f1(pred, batch_y)
            it = it + 1

    accuracy_mean = accuracies/len(val_loader)
    f1_mean = f1_sum / len(val_loader)
    return accuracy_mean, f1_mean


def training(train_loader, val_loader, print_err=True) :
    """ Training over all the epochs of a neurons network model (UNET)"""
    print("\nStart training with {} epochs, batch size = {}, learning rate = {} and weight decay = {} ...".format(NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY))
    if not torch.cuda.is_available() :
        print("\nThings will go much quicker if you enable a GPU in Colab under 'Runtime / Change Runtime Type'")
    else :
        print("\nYou are running the training of the data on a GPU")

    max_f1 = 0

    # Intialisation of the model, criterion, optimizer and scheduler
    model = UNET().to(DEVICE) 
    criterion = IoULoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = None
    # ATTENTION AUSSI DECOMMENTER LE SCHEDULER.STEP LIGNE 120 !!!!
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7, verbose=True) # scheduler reduces learning rate when a metric has stopped improving

    writer = SummaryWriter() # folder location: runs/May04_22-14-54_s-MacBook-Pro.comment/ comment=''

    for epoch in range(NUM_EPOCHS):
        print("\nEpoch : {} / {}".format(epoch, NUM_EPOCHS-1))

        # Training loop
        train_loss, accuracy_train, f1_train = train_func(train_loader, model, epoch, criterion, optimizer)

        # Validation loop
        accuracy_val, f1_val = train_val(val_loader, model, epoch)

        # Updating the scheduler
        #scheduler.step(train_loss) 

        # Printing the training and validation errors
        if (print_err == True) :
            print("\nEpoch {} | Train loss: {:.5f} train f1 score: {:.5f} and train accuracy: {:.5f}".format(epoch, train_loss, f1_train, accuracy_train))
            print("Epoch {} | Validation f1 score : {:.5f} and validation accuracy: {:.5f}".format(epoch, f1_val, accuracy_val))

        # Saving the model with the best f1 score    
        if f1_val > max_f1 :
            max_f1 = copy.deepcopy(f1_val) 
            max_f1_epoch = copy.deepcopy(epoch)
            save_checkpoint(OUTPUT_DIR, epoch, model, optimizer, scheduler)

        # Using summuray writer library to have plots
        if writer is not None :
            writer.add_scalar('Train loss / epoch', train_loss, epoch)
            writer.add_scalars('F1 score / epoch', {'TrainF1':f1_train,'ValF1': f1_val}, epoch)
            writer.add_scalars('Accuracy / epoch', {'TrainAcc': accuracy_train,'ValAcc': accuracy_val}, epoch)

    writer.close()        
    print("\nThe maximum f1 score over all epochs is {} at epoch {}.".format(max_f1, max_f1_epoch))
    print("\nThe best model over all epochs is saved into folder name {} with name parameters.pt".format(OUTPUT_DIR))


def main() :
    # Loading the data
    root_dir = "../data/training/"
    image_dir = root_dir + "images/"
    gt_dir = root_dir + "groundtruth/"
    train_set = ImgDataset(image_dir, gt_dir, split_ratio=0.8, mode="train")
    val_set = ImgDataset(image_dir, gt_dir, split_ratio=0.8, mode="val")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    training(train_loader, val_loader)


if __name__ == "__main__":
    main()
