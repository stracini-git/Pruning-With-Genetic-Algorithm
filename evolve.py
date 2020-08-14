import os

# from numpy import string_

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import tensorflow.keras
import tensorflow as tf
import MyNetworks
import utils
from MaskIndividual import Individual as mi

print("TF version:        ", tf.__version__)
print("TF.keras version:  ", tensorflow.keras.__version__)


def get_session(gpu_fraction=0.80):
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    return tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


tf.compat.v1.keras.backend.set_session(get_session())

GENE_POOL = [0, 1]


def getmasks(net):
    masks = []

    for l in range(1, len(net.layers)):
        if isinstance(net.layers[l].get_weights(), list):
            masks.append([])
            continue

        m = np.ndarray.astype(net.layers[l].get_mask(), np.int8)
        masks.append(m)

    return masks


def PrepareMaskedMLP(dense_arch, myseed, initializer, activation, masktype, trainW, trainM, p1, alpha):
    if myseed == 0:
        myseed = None

    network = MyNetworks.makeMaskedMLP(dense_arch, activation, myseed, initializer, masktype, trainW, trainM, p1, alpha)
    return network


def setupnetwork():
    myseed = 0
    masktype = "mask"
    activation = "relu"
    initializer = "heconstant"
    trainW, trainM = False, True
    alpha = 0
    p1 = 0.5
    nclasses = 10
    dense_arch = [28 * 28, 100, nclasses]

    network = PrepareMaskedMLP(dense_arch, myseed, initializer, activation, masktype, trainW, trainM, p1, alpha)

    network.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
    network.summary()
    return network


def make_population(n, shapeslist):
    population = []

    for _ in range(n):
        gnome = mi.create_gnome_random(shapeslist)
        population.append(mi(gnome))

    return population


def evolve():
    network = setupnetwork()
    data = utils.SetMyData("MNIST", 1)
    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, nclasses = data
    ntrain = 2000
    Xtrain = Xtrain[:ntrain]
    Ytrain = Ytrain[:ntrain]

    masks = getmasks(network)
    shapeslist = [i.shape for i in masks]

    popsize = 100
    population = make_population(popsize, shapeslist)
    maxgenerations = 1000
    generation = 1
    while generation < maxgenerations:
        sorted_population = sorted(population, key=lambda x: x.cal_fitness(network, Xtrain, Ytrain), reverse=True)

        elitepercent = 2
        fitpercent = 17

        # Perform Elitism, that mean 10% of fittest population
        # goes to the next generation
        s = int((elitepercent * popsize) / 100)
        new_generation = sorted_population[:s]

        # From 50% of fittest population, Individuals
        # will mate to produce offspring
        s = int(((100 - elitepercent) * popsize) / 100)
        for _ in range(s):
            parent1 = np.random.choice(sorted_population[:fitpercent])
            parent2 = np.random.choice(sorted_population[:fitpercent])
            child = parent1.mate(parent2)
            new_generation.append(child)

        population = new_generation
        acc = new_generation[0].cal_fitness(network, Xtrain, Ytrain)
        print("Generation: {}\tAccuracy: {:.4f}, population size: {}".format(generation, acc, len(population)))
        generation += 1

    return 0


def main():
    evolve()
    return 0


if __name__ == '__main__':
    main()
