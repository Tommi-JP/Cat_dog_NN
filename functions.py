import os
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import sys

def show_image(images_arr):
    """
    Display a set of images based on user input.

    Parameters:
    images_arr (list): A list of 5 images to be displayed.

    Prompts the user to decide whether or not to show the images.
    """
    u_input = input("Show images? (Y/N)")
    if u_input in ["Y", "y", "Yes", "yes"]:
        fig, axes = plt.subplots(1, 5, figsize=(20, 20))
        axes = axes.flatten()
        for img, ax in zip(images_arr, axes):
            ax.imshow(img)
        plt.tight_layout()
        plt.show()

    elif u_input in ["N", "n", "No", "no"]:
        print("")
    else:
        print("Invalid input")

def model_compiler_summary():
    """
    Creates and compiles a Keras neural network model, asks the user if they want to save
    the model summary in .txt file. Values can be changed if necessary.

    Returns:
        model (tf.keras.Model): Compiled Keras neural network model.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(2)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # ask user if one wants to create summary.txt file
    u_input = input("Do you want to save model summary? (Y/N)")
    if u_input in ["Y", "y", "Yes", "yes"]:
        with open("summary.txt", "w") as file:
            original_stdout = sys.stdout
            sys.stdout = file
            model.summary()
            sys.stdout = original_stdout
    elif u_input in ["N", "n", "No", "no"]:
        print("Summary not saved")
    else:
        print("Invalid input")

    return model


def statistic(history, epochs, file_path_output):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    original_stdout = sys.stdout
    with open(file_path_output, "w") as f:
        sys.stdout = f 
        for epoch in epochs_range:
            print(f"Epoch: {epoch+1}: Accuracy: {acc[epoch]}, Loss: {loss[epoch]}")
    
    sys.stdout = original_stdout