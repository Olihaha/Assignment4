# Assignment 4 - Simpsons Character Face Classification using Neural Networks (self-assigned)
---
## introduction
![ned](extras/flanders.jpg)

Welcome to the wacky world of Simpsonology! In this code, we embark on an exciting adventure to train a neural network capable of distinguishing one Springfield resident from another. That's right, we're building a state-of-the-art system to classify the hilarious and iconic faces of "The Simpsons" characters. You might be reading this wondering, why the simpsons? Well, based on expectations "The Simpsons" characters offer a unique advantage - their faces are relatively easy to differentiate. With convolutional layers as a guide, i hope to be able to unlock the ability to differentiate between Ned Flanders' wholesome smile and Mr. Burns' nefarious scowl. 

## data
The dataset comprises a diverse range of Simpson character face images. in total 20689 images in 39 different folders each after one character. the dataset can be found at https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset. Please note that the creator by accident accidentally copied the entire folder into his list of characters. Meaning that there is another folder inside the folder that is just a copy of the rest. Naturally dont include this. The dataset also includes a bunch of characters with very little images, such as "troy maclure" with 8 images. I have chosen to get rid of some folders due to extremely low amount of images. the following was removed for having super small image amounts. 

Troy mcclure, miss hoover, lionel hutz, gil, fat tony, disco stu, waylon smithers, 

after deleting the following we end up with 34 folders with 20607 images total. 

## model
To tackle this classification problem, we employ a convolutional neural network (CNN) architecture. CNNs are very effective in image recognition due to their ability to learn spatial herachies of features. 
The dataset is divided into training and validation sets before the model is trained in order to assess the effectiveness of the model. To train the model, we use the categorical cross-entropy loss function and the Adam optimizer. In order to avoid overfitting, early halting is used throughout the training phase. We keep an eye on the loss and accuracy metrics on the training and validation sets while training. These metrics give us information about the model's performance and assist us in locating potential improvement areas.


After training, we generate predictions for each sample to assess the model's performance on the validation set. In order to provide a classification report that includes metrics like precision, recall, and F1-score for each class, we translate the categorical predictions back into their respective labels.

We also create a confusion matrix to display the predictions of the model. This matrix offers a thorough analysis of the model's performance, pointing out any trends or incorrect classifications.

## script function
1. importing necessary libraries
2. The hyperparameters for the model are then defined, including batch size, number of classes (Simpsons characters), number of epochs, input shape (image dimensions), and validation split ratio.
3. next, the neural network model is created
4. the data is loaded and preprocessed using the tensorflow ImageDataGenerator
5. 
6. Generates a classification report that includes evaluation metrics for  predictions. This report is saved in the out folder.
7. also generates a confusion matrix and saves it in the out folder.

### how to replicate (fix)
run the setup script provided which does the following : 
1. Creates a virtual environment specifically for the project
2. Activates the venv
3. Installs the required packages from the requiremnets.txt file
4. Runsthe src file.
5. Deactivates the venv

### copy the repository 
git clone XXXXX
make sure to be in correct directory
(cd assignment4-simpsons-faces)

### directory
Make sure that the repository has the following structure and that when running the script you are in the correct directory. 
Make sure that the simpsons_dataset is in the same folder as the script being ran and that the program is being run from teh same directory.


## Results and findings
