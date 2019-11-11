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
from exc3_1 import create_ds, inference_step
import sys
sys.path.append('./Exc_2')
from exc2_1 import create_dataset

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
        self.fc2 = Dense(18, activation='softmax')

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.pool(x)
        x = self.fc2(x)
        return x


ds_test = create_ds()
ds_train = create_dataset(tf.estimator.ModeKeys.TRAIN)
ds_eval = create_dataset(tf.estimator.ModeKeys.EVAL)

batches_per_epoch = 380
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
eval_loss = Mean('eval_loss', dtype=tf.float32)
eval_accuracy = SparseCategoricalAccuracy('eval_accuracy')
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'Exc_3/logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'Exc_3/logs/gradient_tape/' + current_time + '/test'
img_train_log_dir = 'Exc_3/logs/gradient_tape/' + current_time + '/img_train'
img_eval_log_dir = 'Exc_3/logs/gradient_tape/' + current_time + '/img_eval'

data = pd.read_csv('Exc_3/predictions.txt',
                   header=None,
                   names=['img_id', 'img_class'])


@tf.function
def train_step(sample, model):
    with tf.GradientTape() as tape:
        features, labels = sample
        logits = model(features, training=True)

        tr_vars = model.trainable_variables

        batch_loss = loss_object(labels, logits)
        reduced_loss = tf.reduce_sum(batch_loss) / params['batch_size']
        lossL2_layers = tf.add_n([tf.nn.l2_loss(v)
                                  for v in tr_vars]) * params['weight_decay']
        loss = tf.add_n(
            [reduced_loss, lossL2_layers])

    gradients = tape.gradient(loss, tr_vars)
    optimizer.apply_gradients(zip(gradients, tr_vars))

    train_loss(loss)
    train_accuracy(labels, logits)


@tf.function
def eval_step(sample, model):
    features, labels = sample
    logits = model(features)

    tr_vars = model.trainable_variables

    batch_loss = loss_object(labels, logits)
    reduced_loss = tf.reduce_sum(batch_loss) / params['batch_size']
    lossL2_layers = tf.add_n([tf.nn.l2_loss(v)
                              for v in tr_vars]) * params['weight_decay']
    loss = tf.add_n(
        [reduced_loss, lossL2_layers])

    eval_loss(loss)
    eval_accuracy(labels, logits)


def main():
    # TF summary writers for Tensorboard
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    eval_summary_writer = tf.summary.create_file_writer(test_log_dir)
    img_train_summary_writer = tf.summary.create_file_writer(img_train_log_dir)
    img_eval_summary_writer = tf.summary.create_file_writer(img_eval_log_dir)

    model = SimpsonsNet()
    model.build(input_shape=(None, 224, 224, 3))
    model.summary()

    # define checkpoints and checkpoint manager
    ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                               optimizer=optimizer,
                               net=model)
    restore_manager = tf.train.CheckpointManager(
        ckpt, 'Exc_2/tf_ckpts', max_to_keep=3)
    manager = tf.train.CheckpointManager(
        ckpt, 'Exc_3/tf_ckpts_3', max_to_keep=3)
    ckpt.restore(restore_manager.latest_checkpoint)

    # TF summary writers for Tensorboard
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    if restore_manager.latest_checkpoint:
        print('\nRestored from {}'.format(restore_manager.latest_checkpoint))
    else:
        print('\nInitializing from scratch...')

    for epoch in range(params['epochs']):
        for step, sample in enumerate(ds_train):
            train_step(sample, model)
            if step % 100 == 0:
                features = sample[0]
                with img_train_summary_writer.as_default():
                    tf.summary.image('{}_img'.format(step),
                                     features,
                                     step=step,
                                     max_outputs=5)
                    tf.summary.flush()

        with train_summary_writer.as_default():
            tf.summary.scalar('train_loss', train_loss.result(), step=epoch+1)
            tf.summary.scalar('train_accuracy',
                              train_accuracy.result(), step=epoch+1)

        for step, sample in enumerate(ds_eval):
            eval_step(sample, model)
            if step % 100 == 0:
                features = sample[0]
                with img_eval_summary_writer.as_default():
                    tf.summary.image('{}_img'.format(step),
                                     features,
                                     step=step,
                                     max_outputs=5)
                    tf.summary.flush()

        with eval_summary_writer.as_default():
            tf.summary.scalar('eval_loss', eval_loss.result(), step=epoch+1)
            tf.summary.scalar(
                'eval_accuracy', eval_accuracy.result(), step=epoch+1)

        manager.save(checkpoint_number=optimizer.iterations.numpy())

        template = 'Epoch {}/{}, Loss: {}, Acc: {}, Eval Loss: {}, Eval Acc: {}'
        print(template.format(epoch+1,
                              params['epochs'],
                              train_loss.result(),
                              train_accuracy.result()*100,
                              eval_loss.result(),
                              eval_accuracy.result()*100))
        # Reset metrics every epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        eval_loss.reset_states()
        eval_accuracy.reset_states()

    with open('Exc_3/predictions_advanced_model.txt', 'w+') as file:
        for sample in ds_test:
            image, image_ids = sample
            image_ids = np.array(image_ids)
            predictions = inference_step(image, model)

            predicted_classes = np.argmax(predictions, axis=1)
            predicted_classes = predicted_classes.tolist()

            for cls, idx in zip(predicted_classes, image_ids):
                file.write(f'{idx},{cls}\n')


if __name__ == "__main__":
    main()
