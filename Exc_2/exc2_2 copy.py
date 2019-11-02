import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy
from exc2_1 import create_dataset
import os
from datetime import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


params = {
    'learning_rate': 0.001,
    'momentum': 0.9,
    'epochs': 10
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
        self.base_model.trainable = False
        self.pool = GlobalAveragePooling2D()
        self.feature_batch_avarage = self.pool(self.feature_batch)
        self.fc2 = Dense(47, activation='softmax')
        self.prediction_batch = self.fc2(self.feature_batch_avarage)

    def call(self):
        model = tf.keras.Sequential([
            self.base_model,
            self.pool,
            self.fc2
        ])
        return model


optimizer = SGD(learning_rate=params['learning_rate'],
                momentum=params['momentum'])
loss = SparseCategoricalCrossentropy(from_logits=True,
                                     reduction=tf.keras.losses.Reduction.NONE)
train_loss = Mean(name='train_loss')
train_accuracy = SparseCategoricalAccuracy()
eval_loss = Mean(name='eval_loss')
eval_accuracy = SparseCategoricalAccuracy()


for image_batch, label_batch in ds.take(1):
    pass
print(image_batch.shape)

model = SimpsonsNet(image_batch)
model = model.call()
model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy'])
model.summary()


loss0, accuracy0 = model.evaluate(ds_eval, steps=1)
print(loss0, accuracy0)


logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
# image_callback = tf.keras.callbacks.LambdaCallback(
#     n_epoch_end=
# )

def train_step(samples):
    history = model.fit(samples, epochs=1, callbacks=[tensorboard_callback])
    acc = history.history['accuracy']
    loss = history.history['loss']
    print(acc, loss)
    return acc, loss


def eval_step(samples):
    history = model.fit(samples, epochs=30)
    acc = history.history['accuracy']
    loss = history.history['loss']
    print(acc, loss)
    return acc, loss


# for sample in ds:
for i in range(10):
    data = train_step(ds)
    print(data)

# for sample in ds_eval:
#     e_acc, e_loss = eval_step(sample)
