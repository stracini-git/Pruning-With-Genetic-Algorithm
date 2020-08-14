# Training neural network masks with a genetic algorithm
This repository is closely related to 
[SelfPruningNeuralNetworks](https://github.com/stracini-git/SelfPruningNeuralNetworks).
It uses the same networks with masking and self-pruning capacity. However, here the networks are not 
trained with gradient descent. Instead this code is using a population of masks which are 
applied on a randomly initialized network. A genetic algorithm evolves this population of 
masks such that the classification accuracy of the network increases when applying them.


## How to run
Just execute the code:
 ```markdown
~$ python evolve.py

 
TF version:         1.14.0
TF.keras version:   2.2.4-tf

Model: "FC784_100_10_IDbbbf629_SNone"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 784)]             0
_________________________________________________________________
masked_dense (MaskedDense)   (None, 100)               156800
_________________________________________________________________
masked_dense_1 (MaskedDense) (None, 10)                2000
=================================================================
Total params: 158,800
Trainable params: 79,400
Non-trainable params: 79,400
_________________________________________________________________
Generation: 1	Accuracy: 0.1545, population size: 100
Generation: 2	Accuracy: 0.1620, population size: 100
Generation: 3	Accuracy: 0.1740, population size: 100
Generation: 4	Accuracy: 0.1885, population size: 100
Generation: 5	Accuracy: 0.1975, population size: 100
Generation: 6	Accuracy: 0.2110, population size: 100
Generation: 7	Accuracy: 0.2150, population size: 100
Generation: 8	Accuracy: 0.2195, population size: 100
Generation: 9	Accuracy: 0.2280, population size: 100
Generation: 10	Accuracy: 0.2430, population size: 100
Generation: 11	Accuracy: 0.2535, population size: 100
Generation: 12	Accuracy: 0.2535, population size: 100
Generation: 13	Accuracy: 0.2565, population size: 100
Generation: 14	Accuracy: 0.2720, population size: 100
Generation: 15	Accuracy: 0.2720, population size: 100
Generation: 16	Accuracy: 0.2925, population size: 100
Generation: 17	Accuracy: 0.2925, population size: 100
Generation: 18	Accuracy: 0.3015, population size: 100
Generation: 19	Accuracy: 0.3260, population size: 100
Generation: 20	Accuracy: 0.3260, population size: 100
Generation: 21	Accuracy: 0.3260, population size: 100
Generation: 22	Accuracy: 0.3285, population size: 100
Generation: 23	Accuracy: 0.3320, population size: 100
Generation: 24	Accuracy: 0.3370, population size: 100
Generation: 25	Accuracy: 0.3425, population size: 100
Generation: 26	Accuracy: 0.3470, population size: 100
Generation: 27	Accuracy: 0.3530, population size: 100
Generation: 28	Accuracy: 0.3570, population size: 100
Generation: 29	Accuracy: 0.3675, population size: 100
Generation: 30	Accuracy: 0.3725, population size: 100
Generation: 31	Accuracy: 0.3725, population size: 100
Generation: 32	Accuracy: 0.3805, population size: 100
Generation: 33	Accuracy: 0.3805, population size: 100
Generation: 39	Accuracy: 0.3865, population size: 100



```
