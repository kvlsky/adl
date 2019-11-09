import tensorflow as tf
import numpy as np
import sys
sys.path.append('./Exc_2')
from exc2_2 import SimpsonsNet, train_step


def get_tfrecord_dataset(filename):
    raw_dataset = tf.data.TFRecordDataset(filename)
    return raw_dataset


def parse_example(serialized_example):
    feature_description = {
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'image_id': tf.io.FixedLenFeature([], tf.int64),
    }
    raw_ds = tf.io.parse_single_example(serialized_example,
                                        feature_description)

    return raw_ds


def read_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(images=image, size=(224, 224))

    return image


def parse_img(parsed_record):
    # for parsed_record in parsed_dataset:
    image = read_image(parsed_record['image_raw'])
    image_id = parsed_record['image_id']
    return image, image_id


def create_ds():
    raw_dataset = get_tfrecord_dataset('Exc_3/simpsons_test.tfrecord')
    ds = raw_dataset.map(parse_example)
    ds = ds.map(lambda x: parse_img(x))
    ds = ds.batch(16)

    return ds


def inference_step(inputs, model):
    logits = model(inputs, training=False)
    return logits


def main():
    ds = create_ds()
    model = SimpsonsNet()
    model.build(input_shape=(None, 224, 224, 3))
    model.summary()

    counter = 0
    with open('Exc_3/predictions.txt', 'w+') as file:
        for sample in ds:
            counter += 1
            image, image_ids = sample
            image_ids = np.array(image_ids)
            predictions = inference_step(image, model)

            predicted_classes = np.argmax(predictions, axis=1)
            predicted_classes = predicted_classes.tolist()

            for cls, idx in zip(predicted_classes, image_ids):
                file.write(f'{idx},{cls}\n')

            print(f'Batch : {counter}')


if __name__ == "__main__":
    main()
