# PyML

A library containing basic structure and layers for deep learning.

## Introduction

PyML, written in python, is a library that includes basic frameworks and layers for introductory level machine learning, aiming to provide easy and simple solutions to basic networks with clarity as well as comparable speed and accuracy with other popular deep learning frameworks. The layers are designed to be easily combined in order to fit different network.

### File structure:

core
	frame.py: the structure that integrates all layers and performs training
	layers: containing all layers
		Layer.py: superclass for all layers
		DenseLayer.py
		ReLu.py
		Softmax.py
		SVM.py
data
	data_util.py: provides easy access to all datasets contained
	cifar-10
	mnist
examples
	Sample files that perform deep learning using the library
