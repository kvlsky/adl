import tensorflow as tf
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.layers import StackedRNNCells, LSTMCell, Dense, RNN
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Mean
from exc4_1 import get_text_dataset, test_ds
import pickle
import time
import datetime
import os
import numpy as np


class CharRNN(tf.keras.Model):
    def __init__(self):
        super(CharRNN, self).__init__()
        self.lstm_cell = LSTMCell(units=65)
        self.dense_layer = Dense(65)
        self.cells = [
            self.lstm_cell,
            self.lstm_cell,
            self.lstm_cell
        ]
        self.lstm = StackedRNNCells(self.cells)

    def call(self, inp):
        multi_rnn_cell = StackedRNNCells(self.cells)
        zero_state = multi_rnn_cell.get_initial_state(batch_size=50,
                                                      dtype=tf.float32)
        multi_rnn_wrapper = RNN(
            multi_rnn_cell, return_sequences=True, return_state=True)
        res = multi_rnn_wrapper(inp, initial_state=zero_state)[0]
        res = self.dense_layer(res)

        return res

    def inference(self):
        multi_rnn_cell = StackedRNNCells(self.cells)
        initial_state = multi_rnn_cell.get_initial_state(batch_size=50,
                                                         dtype=tf.float32)
        sample_initial_char = test_ds(65)
        state = [initial_state]
        inp = sample_initial_char

        for sample_step in range(50):
            y, state = multi_rnn_cell(inputs=inp, states=state)
            out = self.dense_layer(y)
            inp = test_ds(65)

        return out


EPOCHS = 10
num_chars = 202651
seq_len = 49
batch_size = 50
decay_steps = int(int(num_chars / seq_len) / batch_size)
lr_schedule = ExponentialDecay(
    0.002,
    decay_steps=decay_steps,
    decay_rate=0.97,
    staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
checkpoint_dir = 'Exc_4/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'Exc_4/logs/gradient_tape/' + current_time + '/train'
loss_object = CategoricalCrossentropy(from_logits=True)
train_loss = Mean('train_loss', dtype=tf.float32)

with open('Exc_4/vocabulary.pkl', 'rb') as vocab_file:
    vocab = pickle.load(vocab_file)
    reverse_vocab = {v: k for k, v in vocab.items()}

with open('Exc_4/tinyshakespeare.txt') as text_file:
    text = text_file.read()

dataset = get_text_dataset(text, reverse_vocab)

model = CharRNN()


for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape,
          "# (batch_size, sequence_length, vocab_size)")


@tf.function
def train_step(inp, target):
    with tf.GradientTape() as tape:
        predictions = model(inp)
        batch_loss = loss_object(target, predictions)
        loss = tf.reduce_sum(batch_loss) / 50

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss(loss)


@tf.function
def sample(inp, state):
    o, next_state = model.inference(inp, state)
    softmax = tf.nn.softmax(o)
    argmax = tf.argmax(softmax)
    sampled = tf.one_hot(argmax, 65)
    return sampled


@tf.function
def generate_text(model, char2idx, idx2char):
    predictions = model.inference()

    return predictions


def main(train=True):
    if train is True:
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        for epoch in range(EPOCHS):
            start = time.time()
            # model.reset_states()

            for (batch_n, (inp, target)) in enumerate(dataset):
                train_step(inp, target)
                loss = train_loss.result()

                if batch_n % 50 == 0:
                    template = 'Epoch {} Batch {} Loss {}'
                    print(template.format(epoch+1, batch_n, loss))

            if epoch % 5 == 0:
                model.save_weights(checkpoint_prefix.format(epoch=epoch+1))

            print('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', loss, step=epoch+1)
                tf.summary.scalar('learning_rate', lr_schedule, step=epoch+1)

            train_loss.reset_states()

        model.save_weights(checkpoint_prefix.format(epoch=epoch+1))
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    text_generated = generate_text(model=model,
                                   #    start_string=ds_test,
                                   char2idx=reverse_vocab,
                                   idx2char=vocab)
    print(text_generated)


if __name__ == "__main__":
    main(train=False)
