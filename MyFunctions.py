from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np
import uuid


def a2s(a):
    if len(a) == 0:
        return "0_"

    s = ""
    for i in range(len(a) - 1):
        s += str(a[i]) + "_"

    s += str(a[-1])

    return s


def tf_count(t, val):
    elements_equal_to_value = tf.equal(t, val)
    as_ints = tf.cast(elements_equal_to_value, tf.float32)
    count = tf.reduce_sum(as_ints) / tf.size(t, out_type=tf.dtypes.float32)
    return count


def heconstant(p1, myseed):
    def initializer(shape, dtype=None):
        nlp = np.prod(shape[:-1])
        a = np.sqrt(2 / nlp)
        p2 = 1. - p1
        np.random.seed(myseed)
        distribution = np.random.choice([-1., 1.], shape, p=[p1, p2])
        return tf.Variable(a * distribution, dtype=dtype, name=uuid.uuid4().hex)

    return initializer


def nines(shape, dtype=None):
    # nlp = np.prod(shape[:-1])
    # a = np.sqrt(2 / nlp)
    # p2 = 1. - p1
    # np.random.seed(myseed)
    distribution = np.ones(shape)
    return tf.constant(9 * distribution, dtype=tf.dtypes.float32, name="nines_" + uuid.uuid4().hex)


def binary(p1, myseed):
    def initializer(shape, dtype=None):
        p2 = 1. - p1
        np.random.seed(myseed)
        distribution = np.random.choice([-1, 1], shape, p=[p1, p2])
        return tf.Variable(distribution, dtype=dtype, name=uuid.uuid4().hex)

    return initializer


def ternary(shape, dtype=None):
    nlp = np.prod(shape[:-1])
    a = np.sqrt(2 / nlp)
    p1 = .33
    p2 = .33
    p3 = 1 - p1 - p2
    distribution = np.random.choice([-1., 0, 1.], shape, p=[p1, p2, p3])
    return tf.Variable(a * distribution, dtype=dtype, name=uuid.uuid4().hex)


def quaternary(shape, dtype=None):
    nlp = np.prod(shape[:-1])
    a = np.sqrt(2 / nlp)
    p1 = .25
    p2 = .25
    p3 = .25
    p4 = 1 - p1 - p2 - p3
    distribution = np.random.choice([-2., -1., 1., 2.], shape, p=[p1, p2, p3, p4])
    return tf.Variable(a * distribution, dtype=dtype, name=uuid.uuid4().hex)


def mynormal(shape, dtype=None):
    a = np.sqrt(2 / np.sum(shape))
    name = uuid.uuid4().hex
    return tf.Variable(np.random.normal(0, a, shape), dtype=dtype, name=name)


def myuniform(shape, dtype=None):
    a = 6 * np.sqrt(2 / np.sum(shape))
    name = uuid.uuid4().hex
    return tf.Variable(np.random.uniform(-a, a, shape), dtype=dtype, name=name)


def activate(x, activationtype):
    if 'relu' in activationtype:
        return tf.keras.activations.relu(x)

    if 'softmax' in activationtype:
        return tf.keras.activations.softmax(x)

    if 'sigmoid' in activationtype:
        return tf.keras.activations.sigmoid(x)

    if 'swish' in activationtype:
        return tf.keras.activations.sigmoid(x) * x

    if "elu" in activationtype:
        return tf.keras.activations.elu(x)

    if "selu" in activationtype:
        return tf.keras.activations.selu(x)

    if "flip" in activationtype:
        return flip(x)

    if "mask" in activationtype:
        return mask(x)

    if "linear" in activationtype:
        return x

    if activationtype is None:
        return x

    return x


@tf.custom_gradient
def mask(x):
    y = K.sign(tf.keras.activations.relu(x))

    def grad(dy):
        return dy

    return y, grad


@tf.custom_gradient
def mask_rs(x):
    y = K.sign(tf.keras.activations.relu(x))
    scalefactor = tf.compat.v1.size(y, out_type=tf.dtypes.float32) / (1 + tf.math.count_nonzero(y, dtype=tf.dtypes.float32))
    y *= scalefactor

    def grad(dy):
        return dy

    return y, grad


@tf.custom_gradient
def binarize(x):
    y = K.sign(tf.keras.activations.relu(x))

    def grad(dy):
        return dy

    return y, grad


@tf.custom_gradient
def flip(x):
    # y = K.sign(tf.keras.activations.relu(tf.keras.activations.tanh(x)))
    # y = K.sign(tf.keras.activations.relu(x))
    # y = K.sign(tf.keras.activations.tanh(x))
    y = K.sign(x)

    def grad(dy):
        return dy

    return y, grad


@tf.custom_gradient
def mask_flip(x):
    # y = K.sign(tf.keras.activations.relu(tf.keras.activations.tanh(x)))
    # y = K.sign(tf.keras.activations.relu(x))
    # y = K.sign(tf.keras.activations.tanh(x))
    # y = K.sign(x)
    a = 0.005
    y = K.sign(tf.keras.activations.relu(x - a) - tf.keras.activations.relu(-x - a))

    # y = K.sign(tf.keras.activations.relu(tf.keras.activations.tanh(x) - a) - tf.keras.activations.relu(-tf.keras.activations.tanh(x) - a))

    # scalefactor = tf.compat.v1.size(y, out_type=tf.dtypes.float32) / (1+tf.math.count_nonzero(y, dtype=tf.dtypes.float32))
    # y *= scalefactor

    def grad(dy):
        return dy

    return y, grad
