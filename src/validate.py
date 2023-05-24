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
from sklearn.metrics import confusion_matrix, classification_report
#visualization
import matplotlib.pyplot as plt
import seaborn as sns


#params
batch_size = 128
num_classes = 34
input_shape = (80, 80, 3)  
validation_split = 0.2

model = tf.keras.models.load_model('model.h5')

# Load and preprocess the data using ImageDataGenerator
data_generator = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=validation_split)

# Define the path to the dataset folder
dataset_dir = 'simpsons_dataset'

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

val_data = data_generator.flow_from_directory(
    dataset_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    seed=42,
    shuffle=False
)

# Generate predictions for the validation set
val_predictions = model.predict(val_data)
val_predictions = np.argmax(val_predictions, axis=-1)

y_true = val_data.classes[:len(val_predictions)]
y_pred = val_predictions

y_true = label_encoder.inverse_transform(y_true)
y_pred = label_encoder.inverse_transform(y_pred)

# Classification report and confusion matrix
classification_rep = classification_report(y_true, y_pred)

# confusion matrix with labels
confusion_mat = confusion_matrix(y_true, y_pred, labels=label_encoder.classes_)

# Save classification report and confusion matrix
with open('out/classification_report.txt', 'w') as f:
    f.write(classification_rep)

plt.figure(figsize=(34, 34))
ax = sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')

# Set labels on the sides
ax.set_xticklabels(label_encoder.classes_, rotation=90)
ax.set_yticklabels(label_encoder.classes_, rotation=0)

plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('out/confusion_matrix.png')
plt.close()