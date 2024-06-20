import sys
import os

def image_loader(image_dir):
    train_dir = os.path.join(image_dir, "train")
    val_dir = os.path.join(image_dir, "test")

    cats_train_dir = os.path.join(train_dir, "cat")
    dogss_train_dir = os.path.join(train_dir, "dog")

    cats_val_dir = os.path.join(val_dir, "cat")
    dogs_val_dir = os.path.join(val_dir, "dog")

    print(f"total training images cats:{len(os.listdir(cats_train_dir))}")
    print(f"total training images dogs:{len(os.listdir(dogss_train_dir))}")

    print(f"total validation images cats:{len(os.listdir(cats_val_dir))}")
    print(f"total validation images dogs:{len(os.listdir(dogs_val_dir))}")

    print(f"total training images:{cats_train_dir + dogss_train_dir}")
    print(f"total validation images:{cats_val_dir + dogs_val_dir}")




