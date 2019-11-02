import os
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
import numpy as np

IMG_HEIGHT_MNET = 224
IMG_WIDTH_MNET = 224

IMG_HEIGHT_INCEPTION = 229
IMG_WIDTH_INCEPTION = 229

IMG_SHAPE_MNET = (IMG_WIDTH_MNET, IMG_HEIGHT_MNET, 3)
IMG_SHAPE_INCEPTION = (IMG_WIDTH_INCEPTION, IMG_HEIGHT_INCEPTION, 3)



# define MobileNetV2 architecture and add Dense layer to predict 1000 classes
mobile_net_v2 = MobileNetV2(input_shape=IMG_SHAPE_MNET,
                            include_top=False,
                            pooling="avg",
                            weights='imagenet')
model_mobile_net_v2 = tf.keras.Sequential(
    mobile_net_v2
)
model_mobile_net_v2.add(tf.keras.layers.Dense(1000, activation='softmax'))
model_mobile_net_v2.summary()

model_mobile_net_v2.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['acc'])

# define InceptionV3 architecture and add Dense layer to predict 1000 classes
'''
change weights to imagenet, beacuse i was not able do donwload
them automatically
and downloaded them manually
'''
inception_v3 = InceptionV3(input_shape=IMG_SHAPE_INCEPTION,
                           include_top=False,
                           pooling="avg",
                           weights='inception_v3_weights.h5')
model_inception_v3 = tf.keras.Sequential(
    inception_v3
)
model_inception_v3.add(tf.keras.layers.Dense(1000, activation='softmax'))
model_inception_v3.summary()

model_inception_v3.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['acc'])


def get_labels(filename):
    with open(filename) as file:
        labels_dict = {}
        data = file.read()
        data_list = data.split('\n')
        for i in data_list[:-1]:
            l_list = i.split(':')
            idx, label = l_list[0], l_list[1]
            labels_dict[idx] = label
    return labels_dict


def get_imgs(folder_path):
    all_imgs = os.listdir(folder_path)
    img_paths = []
    for img in all_imgs:
        img_paths.append(folder_path + img)
    return img_paths


def normalize_inception(image_tensor, height, width):
    image = tf.io.read_file(image_tensor)
    img_decoded = tf.image.decode_jpeg(image)
    img_converted = tf.image.convert_image_dtype(img_decoded, tf.float32)
    img_resized = tf.image.resize(img_converted, [height, width])
    img_reshaped = tf.reshape(img_resized, [1, height, width, 3])

    return img_reshaped


def infer_with_mobilenet_v2(image_tensor):
    predictions = model_mobile_net_v2.predict(image_tensor)
    predicted_class = np.argmax(predictions[0], axis=-1)

    return predicted_class


def infer_with_inception_v3(image_tensor):
    predictions = model_inception_v3.predict(image_tensor)
    predicted_class = np.argmax(predictions[0], axis=-1)

    return predicted_class


img_path = 'Exc_1/Assignment1_supplementary/test_images/'
labels = get_labels('Exc_1/Assignment1_supplementary/labels.txt')
imgs = get_imgs(img_path)


for img in imgs:
    normalized_img_mobile_net_v2 = normalize_inception(img,
                                                       IMG_HEIGHT_MNET,
                                                       IMG_WIDTH_MNET)
    normalized_img_inception_v3 = normalize_inception(img,
                                                      IMG_HEIGHT_INCEPTION,
                                                      IMG_WIDTH_INCEPTION)
    predicted_mobile_net_v2 = infer_with_mobilenet_v2(
        normalized_img_mobile_net_v2)
    predicted_inception_v3 = infer_with_inception_v3(
        normalized_img_inception_v3)

    print(f'MobileNetV2: {img}\n{labels[str(predicted_mobile_net_v2)]}')
    print(f'InceptionV3: {img}\n{labels[str(predicted_mobile_net_v2)]}')

    try:
        with open('Exc_1/exc3/predictions.txt', 'a+') as f:
            f.write(f'MobileNetV2: {img}\n{labels[str(predicted_mobile_net_v2)]}\n')
            f.write(f'InceptionV3: {img}\n{labels[str(predicted_mobile_net_v2)]}\n')
    except Exception as e:
        print(f'Exception occured: {e}')
