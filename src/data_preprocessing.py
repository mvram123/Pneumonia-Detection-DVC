import yaml
import argparse

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def read_params(config_path):

    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def pre_processing(config_path):

    config = read_params(config_path=config_path)
    train_path = config['load_data']['train_path']
    test_path = config['load_data']['test_path']

    rescale = config['preprocessing']['Image_Data_Generator']['rescale']
    shear_range = config['preprocessing']['Image_Data_Generator']['shear_range']
    zoom_range = config['preprocessing']['Image_Data_Generator']['zoom_range']
    horizontal_flip = config['preprocessing']['Image_Data_Generator']['horizontal_flip']

    target_size = config['preprocessing']['target_size']
    batch_size = config['preprocessing']['batch_size']
    class_mode = config['preprocessing']['class_mode']

    train_datagen = ImageDataGenerator(rescale=rescale,
                                       shear_range=shear_range,
                                       zoom_range=zoom_range,
                                       horizontal_flip=horizontal_flip)

    test_datagen = ImageDataGenerator(rescale=rescale)

    training_set = train_datagen.flow_from_directory(train_path,
                                                     target_size=target_size,
                                                     batch_size=batch_size,
                                                     class_mode=class_mode)
    test_set = test_datagen.flow_from_directory(test_path,
                                                target_size=target_size,
                                                batch_size=batch_size,
                                                class_mode=class_mode)

    return [training_set, test_set]


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    parsed_args = args.parse_args()
    pre_processing(config_path=parsed_args.config)

