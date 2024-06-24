import os
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import sys

def directory_join(image_dir):
    """
    Create directory paths for training and validation sets for 'cat' and 'dog' images within the given image directory.

    Args:
        image_dir (str): Directory path containing subdirectories 'train' and 'test' with 'cat' and 'dog' images.

    Returns:
        tuple: A tuple containing:
            - train_dir (str): Directory path for training images.
            - val_dir (str): Directory path for validation images.
            - total_train (int): Total number of training images.
            - total_val (int): Total number of validation images.

    Prints:
        Information about the number of images in each category (cats and dogs) for training and validation sets.
    """
    train_dir = os.path.join(image_dir, "train")
    val_dir = os.path.join(image_dir, "test")

    cats_train_dir = os.path.join(train_dir, "cat")
    dogs_train_dir = os.path.join(train_dir, "dog")

    cats_val_dir = os.path.join(val_dir, "cat")
    dogs_val_dir = os.path.join(val_dir, "dog")

    total_train = len(os.listdir(cats_train_dir)) + len(os.listdir(dogs_train_dir))
    total_val = len(os.listdir(cats_val_dir)) + len(os.listdir(dogs_val_dir))

    print(f"Total training images cats: {len(os.listdir(cats_train_dir))}")
    print(f"Total training images dogs: {len(os.listdir(dogs_train_dir))}")

    print(f"Total validation images cats: {len(os.listdir(cats_val_dir))}")
    print(f"Total validation images dogs: {len(os.listdir(dogs_val_dir))}")

    print(f"Total training images: {total_train}")
    print(f"Total validation images: {total_val}")

    return train_dir, val_dir, total_train, total_val



def image_train_generator(batch_size, train_dir, img_shape):
    """
    Generate batches of augmented images for training a machine learning model.

    Args:
        batch_size (int): Number of images per batch.
        train_dir (str): Directory path containing training images.
        img_shape (int): Size to which images are resized (images will be square with dimensions img_shape x img_shape).

    Returns:
        DirectoryIterator: A DirectoryIterator yielding batches of augmented images and their labels (binary).
    """
        # shaping images for training, can be changed if necessary
    image_gen_train = ImageDataGenerator(
        rescale=1./255, # rescales the pixel values of the images by dividing them by 255, which normalizes the pixel values to be between 0 and 1
        rotation_range=30, # randomly rotates the images
        width_shift_range=0.2, # randomly shifts the images horizontally
        height_shift_range=0.2, # and same vertically
        shear_range=0.2, # applies a shearing transformation with a shear intensity
        zoom_range=0.2, # randomly zooms into the images
        horizontal_flip=True, # randomly flips the images horizontally
        fill_mode='nearest' # determines how newly created pixels are filled in when the image is transformed. 'nearest' means that the nearest pixel value will be used to fill in the gaps.
    )

    # training data generator
    train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                         directory=train_dir,
                                                         shuffle=True,
                                                         target_size=(img_shape, img_shape),
                                                         class_mode='binary')
    
    return train_data_gen



def image_validation_generator(batch_size, val_dir, img_shape):
    """
    Generate batches of validation images for evaluating a machine learning model.

    Args:
        batch_size (int): Number of images per batch.
        val_dir (str): Directory path containing validation images.
        img_shape (int): Size to which images are resized (images will be square with dimensions img_shape x img_shape).

    Returns:
        DirectoryIterator: A DirectoryIterator yielding batches of validation images and their labels (binary).
    """
    # shaping images for validation
    image_gen_val = ImageDataGenerator(rescale=1./255)

    val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                     directory=val_dir,
                                                     target_size=(img_shape, img_shape),
                                                     class_mode='binary')
    
    return val_data_gen


def dataset_generator(train_data_gen, val_data_gen, img_shape):
    """
    Converts data generators into TensorFlow Dataset objects.

    Args:
    - train_data_gen (generator function): Generator function for training data.
    - val_data_gen (generator function): Generator function for validation data.
    - img_shape (int): Size of the images in the dataset (assumes square images).

    Returns:
    - train_dataset (tf.data.Dataset): TensorFlow Dataset object for training data,
      containing batches of images and corresponding labels.
    - val_dataset (tf.data.Dataset): TensorFlow Dataset object for validation data,
      containing batches of images and corresponding labels.

    Each dataset is repeated indefinitely (`train_dataset.repeat()`, `val_dataset.repeat()`)
    to allow multiple epochs during training. They are also prefetched (`prefetch(tf.data.AUTOTUNE)`)
    to improve performance by overlapping data preprocessing and model execution.
    """
    # Convert to tf.data.Dataset
    train_dataset = tf.data.Dataset.from_generator(lambda: train_data_gen, output_signature=(
        tf.TensorSpec(shape=(None, img_shape, img_shape, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)))
    train_dataset = train_dataset.repeat().prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_generator(lambda: val_data_gen, output_signature=(
        tf.TensorSpec(shape=(None, img_shape, img_shape, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)))
    val_dataset = val_dataset.repeat().prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset
    


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
    """
    Outputs the training accuracy and loss for each epoch to a specified file.

    Parameters:
    history (History): A Keras History object obtained from the training of a model. 
                       This object contains the accuracy and loss metrics for both 
                       training and validation datasets across all epochs.
    
    epochs (int): The number of epochs the model was trained for.
    
    file_path_output (str): The file path where the epoch-wise accuracy and loss 
                            details will be saved.

    Writes:
    A text file at the specified file path containing the epoch number, 
    training accuracy, and training loss for each epoch.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    original_stdout = sys.stdout
    with open(file_path_output, "w") as f:
        sys.stdout = f 
        for epoch in epochs_range:
            print(f"Epoch: {epoch+1}: Accuracy: {acc[epoch]} "
                    f"Loss: {loss[epoch]} "
                    f"Validation accuracy: {val_acc[epoch]} "
                    f"Validation loss: {val_loss[epoch]}")

    
    sys.stdout = original_stdout

def save_model(model):
    u_input = input("Do you want to save model? (Y/N)")
    if u_input in ["Y", "y", "Yes", "yes"]:
        model.save("./cat_dog.h5")
    elif u_input in ["N", "n", "No", "no"]:
        print("Model is not saved")
    else:
        print("Invalid input")