import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import os, sys
import torch.optim as optim
import copy
from dataset import ImgDataset 
from helpers import *


#Baseline : MLP model with feedforward and backpropagation

def sigmoid(t):
    """Apply sigmoid function on t.
    Args:       t: scalar or numpy array
    Returns:    scalar or numpy array"""

    return 1.0 / (1 + np.exp(-t))


def grad_sigmoid(t):
    """Return the gradient of sigmoid on t.
    Args:       t: scalar or numpy array
    Returns:    scalar or numpy array"""
        
    return sigmoid(t) * (1 - sigmoid(t))

#Model :  
#Input layer : number of neurons comprising that layer is equal to the number of features (columns) in your data --> 3
#Output layer : since NN is a classifier, then it has a single node --> 1
#Hidden layers : 2/3 the size of the input layer + the size of the output layer --> 3
""""
x = np.array of shape (D, ), with D=3
W = {
    "w_1": np.ones((3, 3)),
    "w_2": np.ones(3)
}
y = 1
"""

def simple_feed_forward(x, W):
    """Do feed-forward propagation.
    Args:
        x: numpy array of shape (D, )
        W: a dictionary of numpy array, with two elements, w_1 and w_2.
            w_1: shape=(D, K)
            w_2: shape=(K, )
    Returns:
        z1: a numpy array, generated from the hidden layer (before the sigmoid function) 
        z2: scalar number, generated from the output layer (before the sigmoid function)
        y_hat: a scalar (after the sigmoid function)"""
    
    x_0 = x
    z_1 = W["w_1"].T @ x_0
    x_1 = sigmoid(z_1)
    z_2 = W["w_2"].T @ x_1
    y_hat = sigmoid(z_2)
    
    return z_1, z_2, y_hat


def simple_backpropagation(y, x, W):
    """Do backpropagation and get delta_W.
    Args:
        y: scalar number
        x: numpy array of shape (D, )
        W: a dictionary of numpy array, with two elements, w_1 and w_2.
            w_1: shape=(D, K)
            w_2: shape=(K, )
    Returns:
        grad_W: a dictionary of numpy array. It corresponds to the gradient of weights in W."""
    
    # Feed forward
    z_1, z_2, y_hat = simple_feed_forward(x, W)
    x_1 = sigmoid(z_1)
    # Backpropogation
    delta_2 = (y_hat - y) * grad_sigmoid(z_2)
    delta_w_2 = delta_2 * x_1
    delta_1 = delta_2 * W["w_2"] * grad_sigmoid(z_1)
    delta_w_1 = np.outer(x, delta_1)
    
    return {"w_2": delta_w_2, "w_1": delta_w_1}



def main() :
    # Loading the data
    root_dir = "../data/training/"
    image_dir = root_dir + "images/"
    gt_dir = root_dir + "groundtruth/"
    train_set = ImgDataset(image_dir, gt_dir, split_ratio=0.8, mode="train")
    val_set = ImgDataset(image_dir, gt_dir, split_ratio=0.8, mode="val")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    clf = MLPClassifier(max_iter=1000,
                    hidden_layer_sizes=(3,),
                    random_state=5,
                    verbose=True,
                    learning_rate_init=0.01)
    clf.fit(train_features, train_labels)

    ypred=clf.predict(test_features)
    
    accuracy_score(test_labels, ypred)

if __name__ == "__main__":
    main()




""" ------------------------------ TEST MAIN 2 --------------------------------
def main() :
    # Loading the data
    root_dir = "../data/training/"
    image_dir = root_dir + "images/"
    gt_dir = root_dir + "groundtruth/"
    train_set = ImgDataset(image_dir, gt_dir, split_ratio=0.8, mode="train")
    val_set = ImgDataset(image_dir, gt_dir, split_ratio=0.8, mode="val")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    train_features, train_labels = next(iter(train_loader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    clf = MLPClassifier(max_iter=1000,
                    hidden_layer_sizes=(3,),
                    random_state=5,
                    verbose=True,
                    learning_rate_init=0.01)
    clf.fit(train_features, train_labels)

    
    test_features, test_labels = next(iter(val_loader))
    ypred=clf.predict(test_features)
    
    accuracy_score(test_labels, ypred)
"""




""" ------------------------------ TEST MAIN 3 --------------------------------
def main() :
    # Loading the data
    root_dir = "../data/training/"
    image_dir = root_dir + "images/"
    gt_dir = root_dir + "groundtruth/"
    train_set = ImgDataset(image_dir, gt_dir, split_ratio=0.8, mode="train")
    val_set = ImgDataset(image_dir, gt_dir, split_ratio=0.8, mode="val")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    train_features, train_labels = next(iter(train_loader))
    #img = train_features[0].squeeze()
    #label = train_labels[0]
    print(f"Feature batch shape: {img.size()}")
    print(f"Labels batch shape: {label.size()}")
    
    nsamples, nx, ny = img.shape
    d2_img = img.reshape((nsamples,nx*ny))

    clf = MLPClassifier(max_iter=1000,
                    hidden_layer_sizes=(3,),
                    random_state=5,
                    verbose=True,
                    learning_rate_init=0.01)
    clf.fit(d2_img, label)

    
    test_features, test_labels = next(iter(val_loader))
    img_test = test_features[0].squeeze()
    label_test = test_labels[0]

    nsamples, nx, ny = img_test.shape
    d2_img_test = img_test.reshape((nsamples,nx*ny))
    ypred=clf.predict(d2_img_test)
    
    accuracy_score(label_test, ypred)
"""