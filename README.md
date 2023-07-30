Problem statement: We have our own custom dataset of pizza suchi and steak and we want to make a model to classify them. It is our own customd dataset
We will put our steps into sections for better comprehension
Section 0: (Importing important libraries)
•	We import important libraries

Section 1: Data acquisition
 We download a custom dataset derived from Food101 dataset, parent dataset contains 101 foodsample pictures, our derived dataset contains only three food pictures, pizza,steak and suchi.
•	We download the data from https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip

Section 2:  Data preparation
•	We use walk_through_dir() method to walk throught each and every directory in our downloaded dataset, it will be handy later on 
•	We create traing and testing variables and assign them their reletive paths
•	We visualize the image. We choose a random image from the dataset and print its metadata along with the picture, we use python image library for this purpose (PIL)
•	We can also do the above steps using matplotlib.pyplot.imshow(), but we will have to convert our images to numpy array first
Section 3: Data transformation
•	We want to load our data to pytorch to put in into our models and train those models subsiquently
•	We will convert our images into tensors
•	Turn our images or tensors into dataset and dataloader using torch.utils.data.Dataset and torch.utils.data.DataLoader. We can also convert our images/tensors into dataset and dataloader which will will see later on how
•	We import
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms. These libraries and make our transform and write a function plot_transformed_images to plot orignal image and our transformed image side my side 

Section 4: Loading image data using imagefolder and making datasets
Turning our data into datasets
•	We use
 from torchvision import datasets
datasets.ImageFolder to turn our images into dataset. We give it the file directory address, transform we would like to perform on our data and keep target transform to none ( We don’t want to change our targets/labels
•	We make training_data and transform data
•	By doing this we have gotten .classes and class_to_idx attributes on our training_data and testing_data
•	img, label = train_data[0][0], train_data[0][1]
Our img will have image data, its size, its datatype which is torch.float32
•	We want to print our image using matplotlib, we need to permute its parameters as our image format is CHW (Color, height, width), matplotlib prefers HWC. 

Turning our data into dataloader
Turning our Dataset's into DataLoader's makes them iterable so a model can go through learn the relationships between samples and targets (features and labels).
The higher value num_workers is set to, the more compute power PyTorch will use to load your data.
•	We import dataloader 
from torch.utils.data import DataLoader
•	Dataset parameter to this function is the our transformed data or training_data



Section 5: Loading Image Data with a Custom Dataset
 What is pre-build dataloader and dataset didn’t exist?. So we,re gonna make our own custom dataset and dataloader. Important modules that we are going to need

To create Dataset we use:
1.	Os. Python module for dealing with directories (our data is stored in directories)
2.	Pathlib Each of our image has unique filepaths
3.	Torch for all things Pytorch
4.	PIL image for loading images
5.	torch.utils.data.Dataset to subclass and create our own custom datset
6.	torchvision.transforms to turn our images into tensors.
7.	Various types from Python's typing module to add type hints to our code.
•	We import the libraries above
•	We create a helper function to get class names and class_to_idx attribute. Torchvision.datasets.ImageFolder allowed us to do that 
•	Os.scandir gives us the folder names from a specfic directory

We create a helper function name find_classes to give us classes and class_to_idx attribute which imagefolder gave us
It take image directory as an input and returns list of classes and a tuple of class name (keys )with values (integer)

Now, We create our own Dataset
We create our own Dataset which will have custom transform, will be able to read image paths and take our images and convert them into our custom writeen transforms
1.	We will write a class which will subclass torch.utils.dataset
2.	Initialize our class with target directory and our custom transform
3.	We will create several attributes for our class like paths which will have paths of our target images. 
4.	We create a function to load images from the file and return them. For this we could use PIL or torchvision.io for this purpose
5.	Overide the __len__ method of torch.utils.data.Dataset to return the number of samples in the Dataset
6.	Over the getitem method to return a tuple of tensors and an int value

•	We write our transform function 
 
We are creating a helper function to display the images display_random_images()  to display images
We turn our images into data loader. Our custom Class subclass Dataset from torch.utils so we can use Dataloader
We perform data augmentation so our model has difficulty in training data so it performs better on test data
Section 6: Performing data augmentation
Data augmentation is a technique used on training_data so we can modify our data a little bit, so it is harder for our model to train on it. Its like training a model on blurry data. And if it performs good on it. It means it will perform even better on real world data.
Section 7: Building the model ifself
•	We start by creating our custom transform and load our data for our model_0 under train_dataloader_simple and test_dataloader_simple variables
•	We create TinyVSG model VSG for (Visual geometry group). It is a common model for computer vision problems. It has Convolutional2d layers and relu layers in between them.
•	We create training,testing and train function which has both training and testing loop functions, we also provide train function with no of epochs
•	We plot our model training and test loss

Section 8:  To improve our model performance we add data augmentation in our model training transform so it improves it’s learning ability 

Section 9: We create our new model_1 with data augmentation and plot predictions of both models
