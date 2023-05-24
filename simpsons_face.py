#importing packages
import os
import numpy as np
import tensorflow as tf
import pandas as pd
from PIL import Image
# image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)
# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Conv2D, 
                                     MaxPooling2D)
#split dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#visualization
import matplotlib.pyplot as plt
import seaborn as sns

dir_path = os.path.dirname(os.path.realpath(__file__))

# Params
batch_size = 128
num_classes = 34
epochs = 15
input_shape = (80, 80, 3)  
validation_split = 0.2

# Defining model
model = tf.keras.Sequential()
model.add(Conv2D(96, kernel_size=(4, 4), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(192, kernel_size=(4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Flatten())
model.add(Dense(384, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])


# Load and preprocess the data using ImageDataGenerator
data_generator = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=validation_split)

# Define the path to the dataset folder
dataset_dir = dir_path + '/simpsons_trainset'

# Get the list of character folders
character_folders = os.listdir(dataset_dir)

# Initialize lists to store file paths and labels
image_files = []
labels = []

# Iterate over each character folder
for character_folder in character_folders:
    character_dir = os.path.join(dataset_dir, character_folder)
    if os.path.isdir(character_dir):
        character_label = character_folder  

        # Iterate over image files in the character folder
        for filename in os.listdir(character_dir):
            image_file = os.path.join(character_dir, filename)
            if os.path.isfile(image_file):
                image_files.append(image_file)
                labels.append(character_label)

# Split the dataset into training and validation sets
train_files, val_files, train_labels, val_labels = train_test_split(
    image_files, labels, test_size=validation_split, stratify=labels, random_state=42
)

# Convert labels to numerical format
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
val_labels_encoded = label_encoder.transform(val_labels)

# Convert labels to categorical format
train_labels = tf.keras.utils.to_categorical(train_labels_encoded, num_classes=num_classes)
val_labels = tf.keras.utils.to_categorical(val_labels_encoded, num_classes=num_classes)

# Load and preprocess the data using ImageDataGenerator
train_data = data_generator.flow_from_directory(
    dataset_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    seed=42
)

val_data = data_generator.flow_from_directory(
    dataset_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    seed=42
)


# Train the model
history = model.fit(
    train_data,
    steps_per_epoch=train_data.samples // batch_size,
    validation_data=val_data,
    validation_steps=val_data.samples // batch_size,
    epochs=epochs,
)

# Loss history graph
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(dir_path + '/out/loss_history.png')
plt.close()

# Accuracy graph
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy History')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(dir_path + '/out/accuracy_history.png')
plt.close()

model.save(dir_path + '/model.h5')