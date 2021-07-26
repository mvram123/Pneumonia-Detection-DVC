
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import warnings
warnings.filterwarnings('ignore')


def prediction(img_path, model_path):

    model = load_model(model_path)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    img_data = preprocess_input(x)

    classes = model.predict(img_data)
    result = int(classes[0][0])

    if result == 0:
        output = "Affected By Pneumonia"
    else:
        output = "Normal"

    print(output)
    return output


