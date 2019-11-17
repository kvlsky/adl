import tensorflow as tf
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.layers import StackedRNNCells, LSTMCell, Dense, RNN
from tensorflow.keras.losses import CategoricalCrossentropy
from exc4_1 import get_text_dataset, test_ds
import pickle
import time
import datetime
import os


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

    # def inference(self, inp, state=None):
    #     zero_state = self.lstm_cell.get_initial_state(batch_size=50,
    #                                                   dtype=tf.float32)
    #     states = [zero_state]
    #     o, next_state = self.lstm(inputs=inp, states=states[-1])
    #     o = self.dense_layer(o)
    #     states.append(state)
    #     return o, next_state


EPOCHS = 100
decay_steps = int(int(65 / 50) / 50)
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

with open('Exc_4/vocabulary.pkl', 'rb') as vocab_file:
    vocab = pickle.load(vocab_file)
    reverse_vocab = {v: k for k, v in vocab.items()}

with open('Exc_4/tinyshakespeare.txt') as text_file:
    text = text_file.read()

dataset = get_text_dataset(text, reverse_vocab)
ds_test = test_ds(len(vocab))

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
        loss = tf.reduce_mean(batch_loss)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss


@tf.function
def sample(inp, state):
    o, next_state = model.inference(inp, state)
    softmax = tf.nn.softmax(o)
    argmax = tf.argmax(softmax)
    sampled = tf.one_hot(argmax, 65)
    return sampled


@tf.function
def generate_text(model, start_string, char2idx, idx2char):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 1000

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the word returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(
            predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))


def main():
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    for epoch in range(EPOCHS):
        start = time.time()
        model.reset_states()

        for (batch_n, (inp, target)) in enumerate(dataset):
            loss = train_step(inp, target)

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

        model.save_weights(checkpoint_prefix.format(epoch=epoch+1))

    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    text_generated = generate_text(model=model,
                                   start_string=ds_test,
                                   char2idx=reverse_vocab,
                                   idx2char=vocab)
    print(text_generated)


if __name__ == "__main__":
    main()
