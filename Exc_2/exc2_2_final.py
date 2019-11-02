import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy
from exc2_1 import create_dataset
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


params = {
    'learning_rate': 0.001,
    'momentum': 0.9,
    'epochs': 30
}


ds = create_dataset(tf.estimator.ModeKeys.TRAIN)
ds_eval = create_dataset(tf.estimator.ModeKeys.EVAL)


class SimpsonsNet(tf.keras.Model):
    def __init__(self, image_batch):
        super(SimpsonsNet, self).__init__()
        self.base_model = MobileNetV2(input_shape=(224, 224, 3),
                                      weights='imagenet',
                                      include_top=False)
        self.feature_batch = self.base_model(image_batch)
        # Freeze the convolutional base
        self.base_model.trainable = False
        # Add a classification head
        self.pool = GlobalAveragePooling2D()
        self.feature_batch_avarage = self.pool(self.feature_batch)
        self.fc2 = Dense(47, activation='softmax')
        self.prediction_batch = self.fc2(self.feature_batch_avarage)

    def call(self):
        # Now stack the feature extractor, and two layers
        # using a tf.keras.Sequential model
        model = tf.keras.Sequential([
            self.base_model,
            self.pool,
            self.fc2
        ])
        return model


optimizer = SGD(learning_rate=params['learning_rate'],
                momentum=params['momentum'])
loss_object = SparseCategoricalCrossentropy(from_logits=True,
                                            reduction=tf.keras.losses.Reduction.NONE)
train_loss = Mean(name='train_loss')
train_accuracy = SparseCategoricalAccuracy()
eval_loss = Mean(name='eval_loss')
eval_accuracy = SparseCategoricalAccuracy()

# Inspect a batch of data:
for image_batch, label_batch in ds.take(1):
    pass
print(image_batch.shape)

model = SimpsonsNet(image_batch)
model = model.call()

# We have to  compile the model before training it.
model.compile(optimizer=optimizer,
              loss=loss_object,
              metrics=['accuracy'])
model.summary()

output = model.evaluate(ds_eval, steps=1)
# print(output)


# def train_step(images, labels):
#     with tf.GradientTape() as tape:
#         logits = model(images, training=True)
#         loss_value = loss_object(labels, logits)

#     grads = tape.gradient(loss_value, model.trainable_variables)
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))


# for epoch in range(params["epochs"]):
#     for (batch, (images, labels)) in enumerate(ds):
#         train_step(images, labels)
#         print(f'Train Epoch: {epoch + 1}/{params["epochs"]} Batch {batch}',
#                 end="\r")
#     print('Epoch {} finished'.format(epoch))

def train_step(sample, loss=None):
    with tf.GradientTape() as tape:
        features, labels = sample
        logits = model(features)
        batch_loss = loss(y_pred=logits, y_true=labels)
        loss = tf.reduce_sum(batch_loss) / 16

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(y_true=labels, y_pred=logits)


def eval_step(sample, loss=None):
    features, labels = sample
    logits = model(features)

    batch_loss = loss(y_pred=logits, y_true=labels)
    loss = tf.reduce_sum(batch_loss) / 16

    predictions = tf.nn.softmax(logits)

    eval_loss(loss)
    eval_accuracy(y_true=labels, y_pred=predictions)


# TF summary writer for Tensorboard
train_summary_writer = tf.summary.create_file_writer(
    'Exc_2/tmp/')
# define checkpoints and checkpoint manager
ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                           optimizer=optimizer,
                           net=model)
manager = tf.train.CheckpointManager(ckpt, 'Exc_2/tf_ckpts', max_to_keep=3)
ckpt.restore(manager.latest_checkpoint)


if manager.latest_checkpoint:
    print(f'\nRestored from {manager.latest_checkpoint}')
else:
    print('\nInitializing from scratch...')


with train_summary_writer.as_default():
    for epoch in range(params['epochs']):
        for step, sample in enumerate(ds):
            train_step(sample, loss_object)
            print(f'Train Epoch: {epoch + 1}/{params["epochs"]} Step: {step}',
                  end="\r")
            if step % 100 == 0:
                features = sample[0]
                tf.summary.image(f'{step}_img',
                                 features,
                                 step=step,
                                 max_outputs=5)
                train_summary_writer.flush()

        manager.save(checkpoint_number=optimizer.iterations.numpy())

        for sample in ds_eval:
            eval_step(sample, loss_object)
            print(f'Eval Epoch: {epoch + 1}/{params["epochs"]} Step: {step}',
                  end="\r")
            if step % 100 == 0:
                features = sample[0]
                tf.summary.image(f'{step}_img',
                                 features,
                                 step=step,
                                 max_outputs=5)
                train_summary_writer.flush()

            print(f'Step: {step}, epoch: {epoch+1}/{params["epochs"]}\
                \nLoss: {train_loss.result()}')
            tf.summary.scalar('train_loss', train_loss.result(), step=epoch)
            tf.summary.scalar('eval_loss', eval_loss.result(), step=epoch)
            tf.summary.scalar('train_acc', train_accuracy.result(), step=epoch)
            tf.summary.scalar('eval_acc', eval_accuracy.result(), step=epoch)
            train_summary_writer.flush()

        print(f"Epoch {epoch + 1} summary:\
                \nMean Train Loss: {train_loss.result()}, \
                \nMean Train Acc: {train_accuracy.result()}, \
                \nEval Loss: {eval_loss.result()}, \
                \nEval Acc: {eval_accuracy.result()}")

        train_loss.reset_states()
        train_accuracy.reset_states()

        eval_loss.reset_states()
        eval_accuracy.reset_states()
