# test_tf_keras_import.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Check TensorFlow and Keras versions
print("TensorFlow version:", tf.__version__)

# Create an instance of ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

print("ImageDataGenerator created successfully")
