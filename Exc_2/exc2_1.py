import tensorflow as tf
import os


base_path = 'Exc_2/simpsons_for_students/imgs'
labels = [l for l in os.listdir(base_path)]
labels = sorted(labels)

dict = {}
with open('Exc_2/class_map.txt', 'w+') as f:
    for idx, label in enumerate(labels):
        f.write(f'{idx},{label}\n')
        dict[label] = idx

n_total = sum([len(files) for r, d, files in os.walk(base_path)])
total_train = 0
total_val = 0

with open('Exc_2/train.txt', 'w+') as train_f:
    with open('Exc_2/val.txt', 'w+') as val_f:
        for path, subdirs, files in os.walk('Exc_2/simpsons_for_students/imgs/'):
            for file in files:
                if not file.endswith('.jpg'):
                    files.remove(file)
                else:
                    pass
            class_dir = path.split('/')[-1]
            n_samples = len(files)
            num_train = int(n_samples * 0.9)
            num_val = n_samples - num_train
            total_train += num_train
            total_val += num_val
            print(f'Dir: {class_dir}, N train: {num_train}, N val: {num_val}')
            for name in files[:num_train]:
                file = os.path.join(path, name)
                train_f.write(f'{file},{dict[class_dir]}\n')
            for name in files[num_train:]:
                file = os.path.join(path, name)
                val_f.write(f'{file},{dict[class_dir]}\n')


def read_image(fname, mode):
    image = tf.io.read_file(fname)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    if mode == tf.estimator.ModeKeys.EVAL:
        image = tf.image.resize(images=image, size=(224, 224))
    elif mode == tf.estimator.ModeKeys.TRAIN:
        image = tf.image.resize(images=image, size=(256, 256))
        image = tf.image.random_crop(image, size=[224, 224, 3])
    # image = tf.cast(image, dtype=tf.uint8)
    # image = tf.reshape(image, [1, 224, 224, 3])

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
    tf.io.write_file(f'Exc_2/res_2_1/{name}_result.jpg', conv_img)


ds = create_dataset(tf.estimator.ModeKeys.TRAIN)
for img, label in ds:
    for idx, image in enumerate(img):
        save_img(image, f'{idx}_')
    break
