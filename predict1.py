import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO

dim = 224


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image


def predict(image: Image.Image):
    image = np.asarray(image.resize((dim, dim)), dtype=np.float32)[..., :3]
    image = np.expand_dims(image, 0)
    image = image / 255
    images = np.vstack([image])

    # load model for prediction
    # Load the TFLite model and allocate tensors.
    interpreter1 = tf.lite.Interpreter(model_path="models/main-model1.tflite")
    interpreter1.allocate_tensors()

    # Get input and output tensors.
    input_details1 = interpreter1.get_input_details()
    output_details1 = interpreter1.get_output_details()

    # # Test the model on random input data.
    input_shape = input_details1[0]['shape']
    interpreter1.set_tensor(input_details1[0]['index'], images)
    interpreter1.invoke()
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter1.get_tensor(output_details1[0]['index'])
    return {
        "prediction": str(np.argmax(output_data)),
        "all": str(output_data)
    }


def make_prediction():
    # Test the model on random input data.
    with open('test-images/normal.jpg', 'rb') as f:
        image = f.read()
    image = read_imagefile(image)
    prediction = predict(image)
    print(prediction)


if __name__ == '__main__':
    make_prediction()





