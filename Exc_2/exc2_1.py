import tensorflow as tf
import pandas as pd
import platform

if platform.system() != 'Darwin':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    pass

data = pd.read_csv('Exc_2/annotation.txt', header=None,)
labels = data[5].to_list()
labels = sorted(list(set(labels)))

dict = {}
with open('Exc_2/class_map.txt', 'w+') as f:
    for idx, label in enumerate(labels):
        f.write('{},{}\n'.format(idx, label))
        dict[label] = idx

img_labels = data[[0, 5]]


with open('Exc_2/train.txt', 'w+') as train_f:
    with open('Exc_2/val.txt', 'w+') as val_f:
        files = img_labels[0].to_list()
        labels = img_labels[5].to_list()
        n_samples = len(files)
        num_train = int(n_samples * 0.9)
        num_val = n_samples - num_train
        for file, cls in zip(files[:num_train], labels[:num_train]):
            file = 'Exc_2' + file[1:]
            train_f.write('{},{}\n'.format(file, dict[cls]))
        for file, cls in zip(files[:num_train], labels[:num_train]):
            file = 'Exc_2' + file[1:]
            val_f.write('{},{}\n'.format(file, dict[cls]))


def read_image(fname, mode):
    image = tf.io.read_file(fname)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    if mode == tf.estimator.ModeKeys.EVAL:
        image = tf.image.resize(images=image, size=(224, 224))
    elif mode == tf.estimator.ModeKeys.TRAIN:
        image = tf.image.resize(images=image, size=(256, 256))
        image = tf.image.random_crop(image, size=[224, 224, 3])

    return image


def read_line(line, mode):
    res = tf.io.decode_csv(line, record_defaults=[[""], [0]], field_delim=',')
    img = read_image(res[0], mode)
    return img, res[1]


def create_dataset(mode='all'):
    train_dataset = tf.data.TextLineDataset(['Exc_2/train.txt'])
    val_dataset = tf.data.TextLineDataset(['Exc_2/val.txt'])

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_dataset = train_dataset.shuffle(100)
        train_dataset = train_dataset.map(lambda x: read_line(x, mode))
        train_dataset = train_dataset.batch(16)
        return train_dataset
    elif mode == tf.estimator.ModeKeys.EVAL:
        val_dataset = val_dataset.map(lambda x: read_line(x, mode))
        val_dataset = val_dataset.batch(16)
        return val_dataset


def save_img(img, name):
    conv_img = img
    conv_img = tf.image.convert_image_dtype(conv_img, dtype=tf.uint8)
    conv_img = tf.cast(conv_img, tf.uint8)
    conv_img = tf.io.encode_jpeg(conv_img, quality=100)
    tf.io.write_file('Exc_2/res_img/{}_result.jpg'.format(name), conv_img)


ds = create_dataset(tf.estimator.ModeKeys.TRAIN)
for img, label in ds:
    for idx, image in enumerate(img):
        save_img(image, '{}_'.format(idx))
        print('saved', idx)
    break
