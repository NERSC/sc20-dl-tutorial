# SC20 Deep Learning at Scale Tutorial

This repository contains the example code material for the SC20 tutorial:
*Deep Learning at Scale*.

The example demonstrates *synchronous data-parallel distributed training* of a
convolutional deep neural network implemented in [PyTorch](https://pytorch.org/)
on a standard computer vision problem. In particular, we are training ResNet50
on the [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) dataset to
classify images into 100 classes.

## Links

Presentation slides for the tutorial can be found at:
https://drive.google.com/drive/folders/1-gi1WvfQ6alDOnMwN3JqgNlrQh7MlIQr?usp=sharing

## Installation

If you're running these examples on the Cori GPU system at NERSC, no
installation is needed; you can simply use our provided modules or
shifter containers.

See the [submit.slr](submit.slr) slurm script for a simple example using
an NVIDIA NGC PyTorch container.

## Model, data, and training code overview

The network architecture for our ResNet50 model can be found in
[models/resnet.py](models/resnet.py). Here we have copied the ResNet50
implementation from torchvision and made a few minor adjustments for the
CIFAR dataset (e.g. reducing stride and pooling).

The data pipeline code can be found in
[utils/cifar100\_data\_loader.py](utils/cifar100_data_loader.py).

The basic training logic can be found in [train.py](train.py). We define a
simple Trainer class with methods for training epochs and evaluating.

## Performance profiling and optimization

Documentation in development.

## Distributed GPU training

Documentation in development.
