import tensorflow as tf
import numpy as np
import pickle


def get_text_dataset(text=None,
                     reverse_vocab=None,
                     mode=None,
                     sequence_length=50,
                     batch_size=50):
    char_list = list(text)
    indices_list = []
    for char in char_list:
        indices_list.append(reverse_vocab[char])
    indices_list = list(map(int, indices_list))
    depth = len(reverse_vocab)
    ds = tf.data.Dataset.from_tensor_slices(indices_list)

    ds = ds.map(lambda x: tf.one_hot(x, depth))
    ds = ds.batch(sequence_length, drop_remainder=True)
    ds = ds.map(lambda x: (x[:-1], x[1:]))

    if mode == tf.estimator.ModeKeys.TRAIN:
        ds = ds.shuffle(10000)
    else:
        pass
    ds = ds.batch(batch_size, drop_remainder=True)

    return ds


def test_ds(vocab_size):
    start_id = np.random.randint(low=0, high=vocab_size, size=(1, 1))
    test_ds = tf.data.Dataset.from_tensor_slices(
        tensors=tf.convert_to_tensor(start_id,
                                     dtype=tf.int64))
    test_ds = test_ds.map(map_func=lambda x: tf.one_hot(x, depth=vocab_size))
    return test_ds


if __name__ == "__main__":
    with open('Exc_4/vocabulary.pkl', 'rb') as vocab_file:
        vocab = pickle.load(vocab_file)
        reverse_vocab = {v: k for k, v in vocab.items()}

    with open('Exc_4/tinyshakespeare.txt') as text_file:
        text = text_file.read()

    dataset = get_text_dataset(text, reverse_vocab)