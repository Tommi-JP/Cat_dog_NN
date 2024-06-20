import os
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

def show_image(images_arr):
    """
    Display a set of images based on user input.

    Parameters:
    images_arr (list): A list of 5 images to be displayed.

    Prompts the user to decide whether or not to show the images.
    """
    u_input = input("Show images? (Y/N)")
    if u_input in ["Y", "y", "Yes", "yes"]:
        fig, axes = plt.subplots(1, 5, figsize=(20,20))
        axes = axes.flatten()
        for img, ax in zip(images_arr, axes):
            ax.imshow(img)
        plt.tight_layout()
        plt.show()

    elif u_input in ["N", "n", "No", "no"]:
        print("")
    else:
        print("Invalid input")


def model_compiler():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(2)
        ])
    
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    return model


def model_summary(model):
    u_input = input("Do you want to see model summary? (Y/N)")
    if u_input in ["Y", "y", "Yes", "yes"]:
        model.summary()
    elif u_input in ["N", "n", "No", "no"]:
        print("")
    else:
        print("Invalid input")