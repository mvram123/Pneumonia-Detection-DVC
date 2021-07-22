import os
import yaml
import argparse
import numpy as np
from glob import glob
import warnings
warnings.filterwarnings('ignore')


# Importing Tensorflow and Keras Libraries

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential



from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input


def read_params(config_path):

    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def prediction(config_path, model_path):
    config = read_params(config_path)
    print(model_path)
    model = load_model(model_path)
    img_path = config['prediction']['normal_img_path']

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    img_data = preprocess_input(x)

    classes = model.predict(img_data)
    result = int(classes[0][0])

    if result == 0:
        output = "Person is Affected By PNEUMONIA"
    else:
        output = "Result is Normal"

    print(output)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    parsed_args = args.parse_args()
    model_path = os.path.join('saved_models', 'trail.h5')
    prediction(config_path=parsed_args.config, model_path=model_path)


