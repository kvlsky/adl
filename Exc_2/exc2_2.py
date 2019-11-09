import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy
from exc2_1 import create_dataset
import datetime
import platform

if platform.system() != 'Darwin':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    pass

params = {
    'learning_rate': 0.001,
    'momentum': 0.9,
    'epochs': 30
}

ds = create_dataset(tf.estimator.ModeKeys.TRAIN)
ds_eval = create_dataset(tf.estimator.ModeKeys.EVAL)


class SimpsonsNet(tf.keras.Model):
    def __init__(self):
        super(SimpsonsNet, self).__init__()
        self.base_model = MobileNetV2(input_shape=(224, 224, 3),
                                      weights='imagenet',
                                      include_top=False)
        # Freeze the convolutional base
        self.base_model.trainable = False
        self.pool = GlobalAveragePooling2D()
        # Add classification layer
        self.fc2 = Dense(18, activation='softmax')

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.pool(x)
        x = self.fc2(x)
        return x


optimizer = SGD(learning_rate=params['learning_rate'],
                momentum=params['momentum'])
loss_object = SparseCategoricalCrossentropy()
# Define our metrics
train_loss = Mean('train_loss', dtype=tf.float32)
train_accuracy = SparseCategoricalAccuracy('train_accuracy')
eval_loss = Mean('eval_loss', dtype=tf.float32)
eval_accuracy = SparseCategoricalAccuracy('eval_accuracy')


# Inspect a batch of data:
for image_batch, label_batch in ds.take(1):
    pass
print(image_batch.shape)

model = SimpsonsNet()
model.build(input_shape=(None, 224, 224, 3))
model.summary()


@tf.function
def train_step(sample):
    with tf.GradientTape() as tape:
        features, labels = sample
        logits = model(features, training=True)

        batch_loss = loss_object(labels, logits)
        loss = tf.reduce_sum(batch_loss) / 16

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, logits)


@tf.function
def eval_step(sample):
    features, labels = sample
    logits = model(features)

    batch_loss = loss_object(labels, logits)
    loss = tf.reduce_sum(batch_loss) / 16

    eval_loss(loss)
    eval_accuracy(labels, logits)


# define checkpoints and checkpoint manager
ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                           optimizer=optimizer,
                           net=model)
manager = tf.train.CheckpointManager(ckpt, 'Exc_2/tf_ckpts', max_to_keep=3)
ckpt.restore(manager.latest_checkpoint)

# TF summary writers for Tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
img_train_log_dir = 'logs/gradient_tape/' + current_time + '/img_train'
img_eval_log_dir = 'logs/gradient_tape/' + current_time + '/img_eval'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
eval_summary_writer = tf.summary.create_file_writer(test_log_dir)
img_train_summary_writer = tf.summary.create_file_writer(img_train_log_dir)
img_eval_summary_writer = tf.summary.create_file_writer(img_eval_log_dir)


if manager.latest_checkpoint:
    print('\nRestored from {}'.format(manager.latest_checkpoint))
else:
    print('\nInitializing from scratch...')

# Train for 1 epoch on train_ds
for sample in ds:
    train_step(sample)
print('Trained for 1 epoch')


for epoch in range(params['epochs']):
    for step, sample in enumerate(ds):
        train_step(sample)
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
    manager.save(checkpoint_number=optimizer.iterations.numpy())

    for step, sample in enumerate(ds_eval):
        eval_step(sample)
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
