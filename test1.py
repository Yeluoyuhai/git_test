
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


# TensorFlow, tf.keras and tensorflow_federated
import tensorflow as tf
# from tensorflow import keras
from tensorflow import keras

# import tensorflow_federated as tff
# import tensorflow_federated


# Helper libraries
import numpy as np
# import matplotlib.pyplot as plt
# import functools
import glob
import os
# import PIL
import time
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES' ] ='1'


# tf2.0 不加报错
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Constants
BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 100
noise_dim = 100
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dim])
# seed是【16，100】的随机张量。

# Data

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = tf.input_data.read_data_sets('mnist', one_hot=True)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()


noise=tf.random.normal(shape=train_images.shape, mean=0.0, stddev=50, dtype=tf.dtypes.float32, seed=None, name=None)
drop_noise=tf.nn.dropout(noise, 0.01)

train_images=tf.add(train_images, tf.cast(drop_noise, dtype=tf.dtypes.float32))

plt.figure()
plt.imshow(train_images[0])

plt.grid(False)
plt.show()
train_images= tf.convert_to_tensor(train_images)
train_images=tf.reshape( train_images, (train_images.shape[0], 28, 28, 1))
#train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5   # Normalization
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
test_images = (test_images - 127.5) / 127.5   # Normalization
#
# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
#
# train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
#
# train_images = (train_images - 127.5) / 127.5  # Normalization
# train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
# test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
# test_images = (test_images - 127.5) / 127.5  # Normaliz

# Generator Model  跟wang_zhibo 论文相比， 判别器和生成器卷积核数量不一样
# ，然后没有将他扁平化成一维张量，这个按照别人的程序应该。。。。。可以。
def make_generator_model():
    model = keras.Sequential()

    model.add(keras.layers.Dense( 7 * 7 *256, use_bias=False, input_shape=(100,)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Batch size is not limited
    model.add(keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    model.add \
        (keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)
    return model
# Discriminator Model
def make_discriminator_model():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10))
    return model
generator = make_generator_model()



# discriminator = make_discriminator_model()

# 总的判别模型
malicious_discriminator = make_discriminator_model()
malicious_discriminator.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
# malicious_discriminator.fit(train_images, train_labels, epochs=15, batch_size=3000)
# print(real_output.shape)
# print(real_output)

# test_loss, test_acc = malicious_discriminator.evaluate(test_images, test_labels, verbose=0)
# print('\nTest accuracy:', test_acc, 'Tset loss:', test_loss)



# Cross entropy
cross_entropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True)


# Loss of discriminator
def discriminator_loss(real_output, fake_output, real_labels):
    real_loss = cross_entropy(real_labels, real_output)

    fake_result = np.zeros(256)
    # Attack label
    for i in range(len(fake_result)):
        fake_result[i] = 0
    fake_loss = cross_entropy(fake_result, fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# Loss of generator
def generator_loss(fake_output):
    ideal_result = np.zeros(256)
    # Attack label
    for i in range(len(ideal_result)):

        ideal_result[i] = 5

    return cross_entropy(ideal_result, fake_output)

# Adam optimizer
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Training step
@tf.function
def train_step(images, labels):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        print(generated_images.shape)

        # real output取的是要模仿的那个数字的概率
        real_output = malicious_discriminator(images, training=False)
        fake_output = malicious_discriminator(generated_images, training=False)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output, real_labels = labels)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, malicious_discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients \
        (zip(gradients_of_discriminator, malicious_discriminator.trainable_variables))


# Train
def train(dataset, labels, epochs):
    for epoch in range(epochs):
        start = time.time()
        for i in range(round(60000 /BATCH_SIZE)):

            image_batch = dataset[ i *BATCH_SIZE:min(60000, ( i +1 ) *BATCH_SIZE)]
            labels_batch = labels[ i *BATCH_SIZE:min(60000, ( i +1 ) *BATCH_SIZE)]
            train_step(image_batch, labels_batch)

        generator.save('epoch_{}_generator1.h5'.format(epoch + 1))
        malicious_discriminator.save('epoch_{}_discriminator.h5'.format(epoch + 1))
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time( ) -start))

    # 最后一个epoch结束后生成图片
#  generate_and_save_images(generator, epochs, seed)

# Generate images
# def generate_and_save_images(model, epoch, test_input):
#     # 注意 training 设定为 False
#     # 因此，所有层都在推理模式下运行（batchnorm）。
#     predictions = model(test_input, training=False)
#     fig = plt.figure(figsize=(4,4))
#
#     for i in range(predictions.shape[0]):
#         plt.subplot(4, 4, i+1)
#         plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
#         plt.axis('off')
#
#     plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
#  model.save('generator.h5')


# Train model
train(train_images, train_labels, EPOCHS)
generator.save('generator1.h5')
malicious_discriminator.save('discriminator.h5')
