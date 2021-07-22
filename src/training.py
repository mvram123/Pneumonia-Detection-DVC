import os
import argparse

# Importing Tensorflow and Keras Libraries

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential


from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from data_preprocessing import read_params, pre_processing


def training_model(config_path):

    config = read_params(config_path=config_path)

    input_shape = config['estimators']['VGG16']['params']['input_shape']
    weights = config['estimators']['VGG16']['params']['weights']
    activation = config['estimators']['VGG16']['params']['activation']

    loss = config['estimators']['VGG16']['params']['loss']
    optimizer = config['estimators']['VGG16']['params']['optimizer']
    metrics = config['estimators']['VGG16']['params']['metrics']

    output = config['base']['output']

    training_set = pre_processing(config_path)[0]
    test_set = pre_processing(config_path)[1]

    # Downloading VGG16 Model
    vgg = VGG16(input_shape=input_shape,
                weights=weights,
                include_top=False)

    # Keeping Original Weights
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

    model.fit_generator(training_set,
                        steps_per_epoch=len(training_set),
                        epochs=1,
                        verbose=1,
                        validation_data=test_set,
                        validation_steps=len(test_set))

    model.save(os.path.join('models_saved', 'trail.h5'))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    parsed_args = args.parse_args()
    training_model(config_path=parsed_args.config)






