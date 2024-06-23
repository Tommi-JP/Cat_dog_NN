from functions import *


def main():
    # the directory where the pictures are, can be changed if necessary
    image_dir = "../Cat_Dog_data"
    # batch size and image shape variables, can be changed if necessary
    batch_size = 32
    img_shape = 224
    # epochs
    epochs = 2
    # save outputfile
    file_path_output = "output.txt"

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

    augmented_images = [train_data_gen[0][0][0] for i in range(5)]
    show_image(augmented_images)

    # shaping images for validation
    image_gen_val = ImageDataGenerator(rescale=1./255)

    val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                     directory=val_dir,
                                                     target_size=(img_shape, img_shape),
                                                     class_mode='binary')

    model = model_compiler_summary()

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # last cherckout before training
    u_input = input("Everything looks good? (Y/N)")
    if u_input in ["Y", "y", "Yes", "yes"]:
        print("Training started!")
    elif u_input in ["N", "n", "No", "no"]:
        sys.exit()
    else:
        print("Invalid input")

    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        history = model.fit(
            train_data_gen,
            steps_per_epoch=int(np.ceil(total_train / float(batch_size))),
            epochs=epochs,
            validation_data=val_data_gen,
            validation_steps=int(np.ceil(total_val / float(batch_size)))
        )
      
    statistic(history, epochs, file_path_output)

    save_model(model) 

if __name__ == "__main__":
    main()
