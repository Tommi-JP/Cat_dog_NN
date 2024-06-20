import tensorflow as tf
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from functions import *

# directory where pictures are
image_dir = "../Cat_Dog_data"
image_loader(image_dir)

batch_size = 32
img_shape = 224

