from functions import *

# the directory where the pictures are, can be changed if necessary
image_dir = "../Cat_Dog_data"
# batch size and image shape variables, can be changed if necessary
batch_size = 32
img_shape = 224


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


# shaping images for training, can be changed if necessary
image_gen_train = ImageDataGenerator(
      rescale=1./255, # rescales the pixel values of the images by dividing them by 255, which normalizes the pixel values to be between 0 and 1
      rotation_range=30, # randomly rotates the images
      width_shift_range=0.2, # randomly shifts the images horizontally
      height_shift_range=0.2, # and same verticaly
      shear_range=0.2, # applies a shearing transformation with a shear intensity
      zoom_range=0.2, # randomly zooms into the images
      horizontal_flip=True, # randomly flips the images horizontally
      fill_mode='nearest') # determines how newly created pixels are filled in when the image is transformed. 'nearest' means that the nearest pixel value will be used to fill in the gaps.

train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(img_shape,img_shape),
                                                     class_mode='binary')

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
show_image(augmented_images)


# shaping images for validiation
image_gen_val = ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                 directory=val_dir,
                                                 target_size=(img_shape, img_shape),
                                                 class_mode='binary')

model_compiler()
model_summary()