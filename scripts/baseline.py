import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import f1_score, accuracy_score
from sklearn.neural_network import MLPClassifier

import os, sys
import torch.optim as optim
import copy
################################ HELPERS ####################################
def value_to_class(v):
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df > foreground_threshold:  # road
        return [0, 1]
    else:  # bgrd
        return [1, 0]

def extract_features_2d(img):
    feat_m = np.mean(img)
    feat_v = np.var(img)
    feat = np.append(feat_m, feat_v)
    return feat

def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches



def main() :
    # Loading the data
    root_dir = "../data/training/"
    image_dir = root_dir + "images/"
    gt_dir = root_dir + "groundtruth/"
    test_dir = "../data/test/"

    files = os.listdir(image_dir)
    print("Loading images")
    images = np.asarray([mpimg.imread(image_dir + files[i]) for i in range(len(files))])
    print("Loading groundtruth images")
    gt = np.asarray([mpimg.imread(gt_dir + files[i]) for i in range(len(files))])

    # Use 16x16 patches
    img_patches = [img_crop(images[i], 16, 16) for i in range(len(files))]
    gt_patches = [img_crop(gt[i], 16, 16) for i in range(len(files))]
    # Linearize list of patches
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = np.asarray([value_to_class(np.mean(gt_patches[i])) for i in range(len(gt_patches))])
    # Convert to dense 1-hot representation.
    labels.astype(np.float32)

    # Separate into training and validation set
    n = 0.8 # split ratio for train/val
    N = len(labels)
    idx = np.floor(n*N).astype('int32') # index to stop training set
    X_train = np.asarray([extract_features_2d(img_patches[i]) for i in np.arange(idx)])
    Y_train = np.asarray([labels[i] for i in np.arange(idx)])

    X_val = np.asarray([extract_features_2d(img_patches[i]) for i in np.arange(idx+1, N)])
    Y_val = np.asarray([value_to_class(np.mean(labels[i])) for i in np.arange(idx+1, N)])

    # Balance the training data if needed
    c0 = 0  # bgrd
    c1 = 0  # road
    for i in range(len(Y_train)):
        if Y_train[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print('Number of data points per class before balancing: c0 = ' + str(c0) + ' c1 = ' + str(c1))
    print('Balancing training data...')
    min_c = min(c0, c1)
    idx0 = [i for i, j in enumerate(Y_train) if j[0] == 1]
    idx1 = [i for i, j in enumerate(Y_train) if j[1] == 1]
    new_indices = idx0[0:min_c] + idx1[0:min_c]
    X_train = X_train[new_indices, :]
    Y_train = Y_train[new_indices]
    c0 = 0
    c1 = 0
    for i in range(len(Y_train)):
        if Y_train[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    train_size = Y_train.shape[0]
    print('Number of data points per class after balancing: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    # Fitting the model
    clf = MLPClassifier(max_iter=200,
                    hidden_layer_sizes=(10, 10),
                    random_state=5,
                    verbose=False,
                    learning_rate_init=1e-4,
                    learning_rate = 'adaptive')

    clf.fit(X_train, Y_train)


    # Testing the model with a validation set
    Y_pred = clf.predict(X_val)
    #print('Y val: {} and Y_pred: {}'.format(Y_val, Y_pred))

    acc0 = accuracy_score(Y_val[:,0], Y_pred[:,0])
    acc1 = accuracy_score(Y_val[:,1], Y_pred[:,1])
    f1 = f1_score(Y_val, Y_pred, average = 'micro')

    print('F1 score: {}\t accuracy 0: {}\t acuracy 1: {}'.format(f1, acc0, acc1))



if __name__ == "__main__":
    main()
