from functions import *

def main():
    # the directory where the pictures are
    image_dir = "../Cat_Dog_data"
    # batch size and image shape variables
    batch_size = 32
    img_shape = 224
    # epochs
    epochs = 2
    # save outputfile
    file_path_output = "output.txt"

    train_dir, val_dir, total_train, total_val = directory_join(image_dir)

    train_data_gen = image_train_generator(batch_size, train_dir, img_shape)
    augmented_images = [train_data_gen[0][0][0] for i in range(5)]
    show_image(augmented_images)

    val_data_gen = image_validation_generator(batch_size, val_dir, img_shape)

    # Convert to tf.data.Dataset
    train_dataset = tf.data.Dataset.from_generator(lambda: train_data_gen, output_signature=(
        tf.TensorSpec(shape=(None, img_shape, img_shape, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)))
    train_dataset = train_dataset.repeat().prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_generator(lambda: val_data_gen, output_signature=(
        tf.TensorSpec(shape=(None, img_shape, img_shape, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)))
    val_dataset = val_dataset.repeat().prefetch(tf.data.AUTOTUNE)

    model = model_compiler_summary()

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # last checkout before training
    u_input = input("Everything looks good? (Y/N)")
    if u_input in ["Y", "y", "Yes", "yes"]:
        print("Training started!")
    elif u_input in ["N", "n", "No", "no"]:
        sys.exit()
    else:
        print("Invalid input")

    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        history = model.fit(
            train_dataset,
            steps_per_epoch=int(np.ceil(total_train / float(batch_size))),
            epochs=epochs,
            validation_data=val_dataset,
            validation_steps=int(np.ceil(total_val / float(batch_size)))
        )
      
    statistic(history, epochs, file_path_output)

    save_model(model) 

if __name__ == "__main__":
    main()
