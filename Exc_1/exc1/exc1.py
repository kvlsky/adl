import tensorflow as tf
import numpy as np


IMG_HEIGHT = 224
IMG_WIDTH = 224


@tf.function
def read_img(file_path):
    image = tf.io.read_file(file_path)
    return image


@tf.function
def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=1)
    return img


@tf.function
def reshape_img(img):
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
    img = tf.reshape(img, [1, IMG_WIDTH, IMG_HEIGHT, 1])
    return img


@tf.function
def process_path(file_path):
    img = read_img(file_path)
    img = decode_img(img)
    img = reshape_img(img)
    return img


@tf.function
def conv2d(x, W, strides=1):
    layer = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    return layer


img_processed = process_path('Exc_1/exc1/test.jpg')

k = np.array(([[1, 1, 1], [1, -8, 1], [1, 1, 1]]))
k = tf.constant(k, dtype=tf.float32, name='k')
kernel = tf.reshape(k, [3, 3, 1, 1], name='kernel')

conv = conv2d(img_processed, kernel)
conv_img = conv[-1]
conv_img = tf.image.convert_image_dtype(conv_img, dtype=tf.uint8)
# print(conv_img)

# save img
conv_img = tf.cast(conv_img, tf.uint8)
conv_img = tf.io.encode_jpeg(conv_img, quality=100)
tf.io.write_file('Exc_1/exc1/result.jpg', conv_img)
