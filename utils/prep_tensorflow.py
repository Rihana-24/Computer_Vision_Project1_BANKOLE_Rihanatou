import tensorflow as tf

def get_pretrained_model_tf():
    batch_size = 64
    img_size = (224, 224)  # match your transform resize

    # Load training dataset
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        r'C:/Users/HP-PC/Desktop/Projet_Computer_Vision/data/training',
        label_mode='int',
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
    )

    # Load testing dataset
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        r'C:/Users/HP-PC/Desktop/Projet_Computer_Vision/data/testing',
        label_mode='int',
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
    )

    # Normalize images to [-1, 1], similar to your PyTorch Normalize((0.5), (0.5))
    normalization_layer = tf.keras.layers.Rescaling(scale=1./127.5, offset=-1)

    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

    return train_dataset, test_dataset
