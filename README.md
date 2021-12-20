# CS-433 Road segmentation project

`Team Members` : Julia Wälti, Louis-Alexandre Ongaro, Garance Haefliger

The goal of this project is to detect roads from a satellite images dataset. We trained a Unet, a typical image segmentation algorithm with some additional tools such as data augmentation, regularization, dropout to increase the efficiency of the model. Our best result is 0.000 F1 score on AICrowd (submission ID : ).


## Requirements
All project was implemented with python 3.8. You will also require the following external libraries :
* torch
* torchvision
* sklearn
* skimage
* Pillow
* cv2
* The basic packages : numpy, scipy, os, sys, random, matplotlib, re, copy


## Structure
In the scripts directory we have :
- `'train.py'` : script to train a model and save it
- `'run.py'` : script to run our saved trained model on the test data and get submission file for AICrowd
- `'model.py'` : script that contains the Unet model
- `'dataset.py'` : script to initialize and get the data
- `'data_augmentation.py'` : script to increase our training data set
- `'baseline.py'` : script of baseline model using a patch-wise MLP model
- `'helpers.py'` : script that contains various helper functions used in the above scripts
- `'mask_to_submission.py'` : script with functions that helps to do a submission.csv file for AICrowd
- `'submission_to_mask.py'` : script with functions that can reconstruct the mask from a submission file

The outputs directory is used to store the model parameters after training, the submission and the predicted binary masks after running the stored model on the test set. The data directory contains the zip files of the training and test data. Be careful to unzip these files before running. The Run_colab.ipynb is notebook that you can load in google colab which is platform that provides a free GPU.


## Training and prediction

The current directory for this project is `'your_path/ml-project-2-jlg_project2/scripts'`

If you want to get the results from our pretrained model :
1. You have to unzip the two zip files in directory `'./data/'` ; it is the training and test data.
2. Then, you can download our pretrained model [link](aremplir!!!) ; you have to unzip the files and place it in in the directory `'./outputs/'`
3. Finally, you can run the command : `python3 run.py` in the terminal by taking care that the mentioned current directory. In `'./outputs/'` directory, you will find the `'submission.csv'` files that can be upload on AICrowd to get the F1 score and accuracy on the test set and the predicted binary masks of the test data.

If you want to train the model by yourself :
1. You have to unzip the two zip files in directory `'./data/'` ; it is the training and test data.
2. In the current directory, you can run the command `python3 train.py` in the terminal to get the your trained parameters. In `'./outputs/'`, you will find a `'parameters.pt'` file that contains the parameters of the best model over all the trained epochs.
3. Finally, you can run the command : `python3 run.py` in the terminal by taking care that the mentioned current directory. In `'./outputs/'` directory, you will find the `'submission.csv'` files that can be upload on AICrowd to get the F1 score and accuracy on 
the test set and the predicted binary masks of the test data.
4. If [TensorBoard](https://www.tensorflow.org/tensorboard/) is installed, training losses and validation score can be visualized. To use Tensorboard, run: `'tensorboard --logdir=runs'`

You can also tune a new model by changing the parameters in bold at the beginning of `'train.py'` before running it. 

Please be concern that you have to run our model on a **GPU**, otherwise it will be really time-consuming. If you do not have GPU in your computer, you can use Google Colab which provided a free GPU. In `'./'` you can find a notebook : `'Run_colab.ipynb'` that contains the principal command to run our model on this platform.













