import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy
import pandas as pd
import numpy as np
import datetime
import platform
from exc3_1 import create_ds

if platform.system() != 'Darwin':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    pass

params = {
    'learning_rate': 0.001,
    'momentum': 0.9,
    'lr_decay_factor': 0.5,
    'num_examples_per_epoch': 890,
    'num_epochs_per_decay': 5,
    'batch_size': 16,
    'weight_decay': 0.0005,
    'epochs': 30
}


class SimpsonsNet(tf.keras.Model):
    def __init__(self):
        super(SimpsonsNet, self).__init__()
        self.base_model = MobileNetV2(input_shape=(224, 224, 3),
                                      weights='imagenet',
                                      include_top=False)
        self.base_model.trainable = True
        self.pool = GlobalAveragePooling2D()
        # Add classification layer
        self.fc2 = Dense(18, activation='softmax')

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.pool(x)
        x = self.fc2(x)
        return x


ds = create_ds()

batches_per_epoch = 56
decay_steps = batches_per_epoch * params['num_epochs_per_decay']
lr_schedule = ExponentialDecay(
    params['learning_rate'],
    decay_steps=decay_steps,
    decay_rate=params['lr_decay_factor'],
    staircase=True)
optimizer = SGD(learning_rate=lr_schedule,
                momentum=params['momentum'])
loss_object = SparseCategoricalCrossentropy()

# Define our metrics
train_loss = Mean('train_loss', dtype=tf.float32)
train_accuracy = SparseCategoricalAccuracy('train_accuracy')
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'Exc_3/logs/gradient_tape/' + current_time + '/train'

data = pd.read_csv('Exc_3/predictions.txt',
                   header=None,
                   names=['img_id', 'img_class'])


# @tf.function
def train_step(sample, model):
    with tf.GradientTape() as tape:
        images, img_ids = sample
        labels = []
        for img_id in np.array(img_ids):
            row = data[data['img_id'] == img_id]
            label = row.iloc[0]['img_class']
            labels.append(label)

        labels = tf.convert_to_tensor(labels, dtype=tf.float32)
        logits = model(images, training=True)

        tr_vars = model.trainable_variables

        batch_loss = loss_object(labels, logits)
        reduced_loss = tf.reduce_sum(batch_loss) / params['batch_size']
        lossL2_layers = tf.add_n([tf.nn.l2_loss(v)
                                  for v in tr_vars]) * params['weight_decay']
        loss = tf.add_n(
            [reduced_loss, lossL2_layers])

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, logits)


def main():
    model = SimpsonsNet()
    model.build(input_shape=(None, 224, 224, 3))
    model.summary()

    counter = 0
    for sample in ds:
        counter += 1
        image = sample[0]
        model(image)
    print(f'Batches per epoch: {counter}')

    # define checkpoints and checkpoint manager
    ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                               optimizer=optimizer,
                               net=model)
    restore_manager = tf.train.CheckpointManager(
        ckpt, 'Exc_2/tf_ckpts')
    manager = tf.train.CheckpointManager(
        ckpt, 'Exc_3/tf_ckpts', max_to_keep=3)
    ckpt.restore(restore_manager.latest_checkpoint)

    # TF summary writers for Tensorboard
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    if restore_manager.latest_checkpoint:
        print('\nRestored from {}'.format(manager.latest_checkpoint))
    else:
        print('\nInitializing from scratch...')

    for epoch in range(params['epochs']):
        for step, sample in enumerate(ds):
            train_step(sample, model)
            print(f'Epoch: {epoch+1}/{params["epochs"]}, Step: {step}')

        with train_summary_writer.as_default():
            tf.summary.scalar('train_loss', train_loss.result(), step=epoch+1)
            tf.summary.scalar('train_accuracy',
                              train_accuracy.result(), step=epoch+1)
        manager.save(checkpoint_number=optimizer.iterations.numpy())

        # Reset metrics every epoch
        train_loss.reset_states()
        train_accuracy.reset_states()


if __name__ == "__main__":
    main()
