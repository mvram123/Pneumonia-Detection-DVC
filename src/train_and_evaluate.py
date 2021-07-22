import os
import yaml
import numpy as np
from glob import glob
import argparse

# Importing Tensorflow and Keras Libraries

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential


from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input


def read_params(config_path):

    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def training_model(config_path):

    config = read_params(config_path=config_path)
    image_size = config['base']['image_size']
    train_path = config['load_data']['train_path']
    test_path = config['load_data']['test_path']
    input_shape = config['estimators']['VGG16']['params']['input_shape']
    weights = config['estimators']['VGG16']['params']['weights']
    output = config['base']['output']
    activation = config['estimators']['VGG16']['params']['activation']
    loss = config['estimators']['VGG16']['params']['loss']
    optimizer = config['estimators']['VGG16']['params']['optimizer']
    metrics = config['estimators']['VGG16']['params']['metrics']

    vgg = VGG16(input_shape=input_shape,
                weights=weights,
                include_top=False)

    for layer in vgg.layers:
        layer.trainable = False

    x = Flatten()(vgg.output)
    prediction = Dense(output, activation=activation)(x)
    model = Model(inputs=vgg.input, outputs=prediction)

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=[metrics]
    )

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory(train_path,
                                                     target_size=(224, 224),
                                                     batch_size=10,
                                                     class_mode='categorical')
    test_set = test_datagen.flow_from_directory(test_path,
                                                target_size=(224, 224),
                                                batch_size=10,
                                                class_mode='categorical')
    print(model.summary())

    history = model.fit_generator(
        training_set,
        validation_data=test_set,
        epochs=1,
        steps_per_epoch=len(training_set),
        validation_steps=len(test_set)
    )

    model.save(os.path.join('models_saved', 'trail.h5'))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    parsed_args = args.parse_args()
    training_model(config_path=parsed_args.config)






