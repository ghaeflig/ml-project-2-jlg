# CS-433 Road segmentation project

`Team Members` : Julia Wälti, Louis-Alexandre Ongaro, Garance Haefliger

The goal of this project is to detect roads from a satellite images dataset. We trained a Unet, a typical image segmentation algorithm with some additional tools such as data augmentation, regularization, dropout to increase the efficiency of the model. Our best result is 0.872 F1 score with an accuracy of 0.927 on AICrowd (submission ID : 169096).


## Requirements
All project was implemented with python 3.8 and was run with a free GPU memory provided by Google colab (Nvidia Tesla V100 GPU). You will also require the following external libraries :
* torch
* torchvision
* sklearn
* skimage
* Pillow
* cv2
* tensorboard (torch.utils.tensorboard)
* The basic packages : numpy, scipy, os, sys, random, matplotlib, re, copy, matplotlib


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

The outputs directory is used to store the model parameters after training, the submission and the predicted binary masks after running the stored model on the test set. The data directory contains the zip files of the training and test data. Be careful to unzip these files before running. The Run_colab.ipynb is a notebook that you can load in Google Colab which is a platform that provides a free GPU.


## Training and prediction

The current directory for this project is `'your_path/ml-project-2-jlg_project2/scripts'`

If you want to get the results from our pretrained model :
1. You have to unzip the two zip files in the directory `'./data/'` ; it is the training and test data.
2. Then, you can download our pretrained model [link](https://drive.google.com/file/d/1tCJs0LqV1BapljetbS_ztGNu2VPYrDZE/view?usp=sharing) ; you have to unzip the file whose name is `'parameters.pt'` and place it in in the directory `'./outputs/'`
3. Finally, you can run the command : `python3 run.py` in the terminal. In `'./outputs/'` directory, you will find:
	* the `'submission.csv'` file that can be uploaded on AICrowd to get the F1 score and accuracy on the test set
	* the predicted binary masks of the test data. 
With this pretrained model, you should obtain exactly our best result by submitting the `'submission.csv'` file to AICrowd

If you want to train the model by yourself :
1. You have to unzip the two zip files in the directory `'./data/'` ; it is the training and test data.
2. In the current directory, you can run the command `python3 train.py` in the terminal to get your trained parameters. In `'./outputs/'`, you will find a `'parameters.pt'` file that contains the parameters of the best model over all the trained epochs.
3. Finally, you can run the command : `python3 run.py` in the terminal. In `'./outputs/'` directory, you will find:
	* the `'submission.csv'` files that can be uploaded on AICrowd to get the F1 score and accuracy on the test set 
	* the predicted binary masks of the test data.
4. If [TensorBoard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html) is installed, training loss/scores and validation scores can be visualized. To use Tensorboard, run: `tensorboard --logdir=runs`
5. If you want to retrain a model with the augmented data, you can do it : `python3 train.py`
6. If you want to retrain the model only on the original data, you have to delete the current training folder in `'./data/'` and unzip the given training.zip. Then, you have to set `LOAD_DATA_AUG` to false at the beginning of the `'train.py'` file and after that, you can launch the command : `python3 train.py`.

You can also tune a new model by changing the parameters in bold at the beginning of `'train.py'` before running it. 

Please be concerned that you have to run our model on a **GPU**, otherwise, it will be really time-consuming. If you do not have GPU on your computer, you can use Google Colab which provided a free GPU. In `'./'` you can find a notebook : `'Run_colab.ipynb'` that contains the principal command to run our model on this platform.









