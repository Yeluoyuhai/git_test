
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# TensorFlow, tf.keras and tensorflow_federated
import tensorflow as tf
#tf.disable_v2_behavior()

# from tensorflow import keras
from tensorflow import keras

# import tensorflow_federated as tff
# import tensorflow_federated


# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import functools
import glob
import os
import PIL
import time
import os
import functools
import numpy as np
import time
from tensorflow.python.ops import array_ops
#import contrib
# tf2.0 不加报错
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Constants
BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 5
noise_dim = 100
num_examples_to_generate = 16

# seed是【16，100】的随机张量。

tfgan =tf.contrib.gan

session=tf.InteractiveSession()
# Data

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = tf.input_data.read_data_sets('mnist', one_hot=True)
train_img=np.empty((45,28,28), dtype = int, order = 'C')
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
j=0
for i in range (20000):
    if (train_labels[i]==5):

        train_img[j]=train_images[i]
        j=j+1
    if (train_labels[i]==9):

        train_img[j]=train_images[i]
        j=j+1
    if(j==45):
        break
print(train_images[1])
print(train_img.shape)



BATCH_SIZE = 64
INCEPTION_URL = 'http://download.tensorflow.org/models/frozen_inception_v1_2015_12_05.tar.gz'
INCEPTION_FROZEN_GRAPH = 'inceptionv1_for_inception_score.pb'

# Run images through Inception.
inception_images = tf.placeholder(tf.float32, [None, 3, None, None])
def inception_logits(images = inception_images, num_splits = 1):
    images = tf.transpose(images, [0, 2, 3, 1])
    size = 299
    images = tf.image.resize_bilinear(images, [size, size])
    generated_images_list = array_ops.split(images, num_or_size_splits = num_splits)
    logits = tf.map_fn(
        fn = functools.partial(
             tfgan.eval.run_inception,
             default_graph_def_fn = functools.partial(
             tfgan.eval.get_graph_def_from_url_tarball,
             INCEPTION_URL,
             INCEPTION_FROZEN_GRAPH,
             os.path.basename(INCEPTION_URL)),
             output_tensor = 'logits:0'),
        elems = array_ops.stack(generated_images_list),
        parallel_iterations = 8,
        back_prop = False,
        swap_memory = True,
        name = 'RunClassifier')
    logits = array_ops.concat(array_ops.unstack(logits), 0)
    return logits

logits=inception_logits()

def get_inception_probs(inps):
    n_batches = int(np.ceil(float(inps.shape[0]) / BATCH_SIZE))
    preds = np.zeros([inps.shape[0], 1000], dtype = np.float32)
    for i in range(n_batches):
        inp = inps[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] / 255. * 2 - 1
        preds[i * BATCH_SIZE : i * BATCH_SIZE + min(BATCH_SIZE, inp.shape[0])] = session.run(logits,{inception_images: inp})[:, :1000]
    preds = np.exp(preds) / np.sum(np.exp(preds), 1, keepdims=True)
    return preds

def preds2score(preds, splits=10):
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

def get_inception_score(images, splits=10):
    assert(type(images) == np.ndarray)
 #   assert(len(images.shape) == 4)
 #   assert(images.shape[1] == 3)
    print(images[1])
# assert(np.min(images[1]) >= 0 and np.max(images[1]) > 10), 'Image values should be in the range [0, 255]'
    print('Calculating Inception Score with %i images in %i splits' % (images.shape[0], splits))
    start_time=time.time()
    preds = get_inception_probs(images)
    mean, std = preds2score(preds, splits)
    print('Inception Score calculation time: %f s' % (time.time() - start_time))
    return mean, std  # Reference values: 11.38 for 50000 CIFAR-10 training set images, or mean=11.31, std=0.10 if in 10 splits.











train_img=train_img.reshape(15,3,28,28)


#generate_and_save_images(generator,1, seed)

a,b=get_inception_score(train_img, splits=2)
print('a=',a,'b=',b)


x=np.load("prediction.npy")
x=x*127.5+127.5
x=x.reshape(16,3,28,28)
a2,b2=get_inception_score(x, splits=1)
print('a2=',a2,'b2=',b2)


x2=np.load("prediction2.npy")

x2=x2*127.5+127.5
x2=x2.reshape(16,3,28,28)
a4,b4=get_inception_score(x2, splits=2)
print('a4=',a4,'b4=',b4)

x1=np.load("prediction1.npy")
x1=x1*127.5+127.5
x1=x1.reshape(16,3,28,28)
a3,b3=get_inception_score(x1, splits=1)
print('a3=',a3,'b3=',b3)

x2=np.load("prediction2.npy")

x2=x2*127.5+127.5
x2=x2.reshape(16,3,28,28)
a4,b4=get_inception_score(x2, splits=2)
print('a4=',a4,'b4=',b4)