#!/usr/bin/env python
# coding: utf-8

# In[8]:


#import the libraries
import numpy as np
import tensorflow as tf
import struct
from tensorflow.keras import models
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

#to make a GUI for the hand writing
from tkinter import *
#image processing
from PIL import Image, ImageOps, ImageGrab


# In[9]:


# Load the dataset
def load_images(file_path):
    with open(file_path, 'rb') as file:
        # Read magic number, number of images, rows, and columns
        magic, num_images, rows, cols = struct.unpack('>IIII', file.read(16))
        # Read the image data and reshape it
        images = np.fromfile(file, dtype=np.uint8).reshape(num_images, rows, cols, 1)
        return images / 255.0  # Normalize pixel values

def load_labels(file_path):
    with open(file_path, 'rb') as file:
        # Read magic number and number of labels
        magic, num_labels = struct.unpack('>II', file.read(8))
        # Read the label data
        labels = np.fromfile(file, dtype=np.uint8)
        return to_categorical(labels, 10)  # One-hot encode the labels


# In[10]:


base_path = r"C:\My files\academic\DTU\Acdemics\DTU_SEMESTER 3\Ercan-Computer vision\Handwritten example"

# Load training and test data using the modified path
train_images = load_images(f"{base_path}\\train-images.idx3-ubyte")
train_labels = load_labels(f"{base_path}\\train-labels.idx1-ubyte")
test_images = load_images(f"{base_path}\\t10k-images.idx3-ubyte")
test_labels = load_labels(f"{base_path}\\t10k-labels.idx1-ubyte")


# In[11]:


# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# In[12]:


# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[13]:


# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.1)


# In[14]:


# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")


# In[ ]:




