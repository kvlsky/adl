import tensorflow as tf
import numpy as np
import os


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
def my_tf_conv2d(image_tensor, kernel_tensor):
    layer = tf.nn.conv2d(image_tensor, kernel_tensor,
                         strides=[1, 1, 1, 1], padding='SAME')
    return layer


w_init = tf.random_normal_initializer()
w = tf.Variable(initial_value=w_init(shape=(3, 3),
                                     dtype='float32'),
                trainable=True)
w = tf.reshape(w, [3, 3, 1, 1])
w = tf.Variable(initial_value=w, trainable=True)

k = np.array(([[1, 1, 1], [1, -8, 1], [1, 1, 1]]))
k = tf.constant(k, dtype=tf.float32, name='k')
constant_kernel = tf.reshape(k, [3, 3, 1, 1])


mse = tf.losses.MeanSquaredError()
optimizer = tf.optimizers.SGD(learning_rate=0.1)


def save_img(img, name):
    conv_img = img[-1]
    conv_img = tf.image.convert_image_dtype(conv_img, dtype=tf.uint8)
    conv_img = tf.cast(conv_img, tf.uint8)
    conv_img = tf.io.encode_jpeg(conv_img, quality=100)
    tf.io.write_file(f'Exc_1/exc2/{name}_result.jpg', conv_img)


@tf.function
def train_step(image):
    with tf.GradientTape() as tape:
        groundtruth = my_tf_conv2d(image, constant_kernel)
        prediction = my_tf_conv2d(image, w)
        loss = mse(groundtruth, prediction)

    var = [w]
    gradients = tape.gradient(loss, var)
    optimizer.apply_gradients(zip(gradients, var))

    return loss, groundtruth, prediction


def get_imgs(folder_path):
    all_imgs = os.listdir(folder_path)
    img_paths = []
    for img in all_imgs:
        img_paths.append(folder_path + img)
    return img_paths


for img in get_imgs('Exc_1/Assignment1_supplementary/test_images/'):
    processed_image = process_path(img)
    for i in range(100):
        loss, g, p = train_step(processed_image)
        print(f'step: {i}, loss: {loss}')
    save_img(g, 'groundtruth')
    save_img(p, 'prediction')


'''
    e) There is almost no difference between prediction and groundtruth.
    There is difference between the Laplacian Filter and the one we learned,
    because loss is not converged to 0 when useing gradient descent.
    To solve this promblem, we can decrease learning rate,
    but it will cause long execution time or use exponential decay.
    From another hand, we can use another optimizer such as Adam, 
    which is higly popular now.

'''
