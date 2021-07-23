import os
import time
from datetime import datetime
import argparse
from tensorflow import keras
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from data_preprocessing import read_params, pre_processing


class TimeHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def training_model(config_path):

    config = read_params(config_path=config_path)

    input_shape = config['estimators']['VGG16']['params']['input_shape']
    weights = config['estimators']['VGG16']['params']['weights']
    activation = config['estimators']['VGG16']['params']['activation']

    loss = config['estimators']['VGG16']['params']['loss']
    optimizer = config['estimators']['VGG16']['params']['optimizer']
    metrics = config['estimators']['VGG16']['params']['metrics']

    model_summary_path = config['estimators']['store']['model_summary_path']
    metrics_path = config['estimators']['store']['metrics_path']

    model_dir = config['model_dir']

    output = config['base']['output']

    # train_data, test_data = pre_processing(config_path=config_path)

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

    # Current datetime
    dt_string = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    file_name = 'model_sum_'+ dt_string + '.txt'

    # Saving Model Summary in a file
    with open(os.path.join(model_summary_path, file_name), 'w') as f:
        model.summary(print_fn=lambda a: f.write(a + '\n'))

    training_set, test_set = pre_processing(config_path=config_path)

    time_callback = TimeHistory()
    history = model.fit(training_set,
                        validation_data=test_set,
                        epochs=1,
                        callbacks=[time_callback],
                        steps_per_epoch=len(training_set),
                        validation_steps=len(test_set))

    time_taken_for_execution = sum(time_callback.times)

    print(f'time taken : {time_taken_for_execution}')

    # Saving Model
    model_name = 'model_' + dt_string + '.h5'
    model.save(os.path.join(model_dir, model_name))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    parsed_args = args.parse_args()
    training_model(config_path=parsed_args.config)






