Neural Network Pruning on CIFAR-10 using VGG Network

This repository contains a hands-on implementation of network pruning techniques—fine-grained pruning and channel pruning—applied to a classical neural network model (VGG) trained on the CIFAR-10 dataset. The primary goal is to reduce both model size and latency while maintaining an acceptable level of performance, showcasing the tradeoffs between accuracy and model efficiency.

Overview

Pruning is a model optimization technique that reduces the number of parameters in a neural network by removing unnecessary weights, making the model more lightweight and efficient. In this project, we focus on two types of pruning:

	•	Fine-Grained Pruning: Pruning individual weights in the network to reduce model size.
	•	Channel Pruning: Pruning entire channels in the convolutional layers, which can significantly reduce the computational cost and inference time.

Key Objectives

	•	Understand the Basics of Pruning: Gain insights into how neural network pruning works and its impact on model size and performance.
	•	Fine-Grained Pruning: Apply pruning on individual weights to reduce model complexity while maintaining accuracy.
	•	Channel Pruning: Prune channels from the convolutional layers to achieve faster inference and reduced computational cost.
	•	Evaluate Performance Improvements: Measure the tradeoffs between accuracy, model size, and speedup after pruning.
	•	Compare Pruning Approaches: Understand the differences and tradeoffs between fine-grained and channel pruning.

Contents

The notebook in this repository is divided into two main sections:

	1.	Fine-Grained Pruning:
	•	Pruning individual weights in the network based on their magnitude.
	•	Evaluation of the pruned model on the CIFAR-10 dataset.
	•	Analysis of accuracy tradeoffs and memory savings after pruning.
	2.	Channel Pruning:
	•	Pruning entire channels from the VGG convolutional layers.
	•	Fine-tuning the model to regain lost accuracy.
	•	Measurement of latency and performance speedup after pruning.

 Installation

To run the notebook locally, follow these steps:

Clone the Repository
!git clone https://github.com/elprofessor-15/Network_pruning_Fine-Grained_and_Channel_Pruning.git
Install Dependencies

You can install the required libraries using the following command:
!pip install torch torchvision matplotlib numpy

Make sure you have Python 3.8+ and PyTorch installed.

Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 50,000 training images and 10,000 test images. The notebook automatically downloads the dataset using the PyTorch torchvision.datasets module.

Pruning Techniques

1. Fine-Grained Pruning

Fine-grained pruning refers to removing individual weights from the network based on their magnitude. The lower magnitude weights contribute less to the model’s performance and are pruned to reduce model size.

In the notebook, we:

	•	Apply pruning to the VGG network.
	•	Remove weights below a specified threshold.
	•	Evaluate the performance and memory reduction after pruning.

2. Channel Pruning

Channel pruning involves pruning entire channels (filters) in the convolutional layers. This method is computationally efficient because it reduces the number of filters in each layer, directly impacting inference speed.

In the notebook, we:

	•	Identify the least important channels.
	•	Prune these channels from the convolutional layers.
	•	Fine-tune the pruned model to recover accuracy.
	•	Measure the speedup and latency improvement.

Usage

Running the Pruning Notebook

After cloning the repository and installing dependencies, you can run the pruning notebook using Jupyter:
jupyter notebook pruning_CIFAR10_VGG.ipynb

The notebook provides step-by-step instructions for applying fine-grained and channel pruning on the VGG network.

Results

	•	Fine-Grained Pruning: Successfully reduces the model size by pruning individual weights while maintaining competitive accuracy on the CIFAR-10 dataset.
	•	Channel Pruning: Achieves significant speedup in inference time by pruning channels, with a minor drop in accuracy that can be recovered through fine-tuning.
	•	Detailed analysis and visualizations of the accuracy vs. sparsity tradeoff are included in the notebook.

Sample Results:

	•	Original VGG Model:
	•	Accuracy: ~93%
	•	Model Size: 14.7 MB
	•	Inference Time: 2.5 ms
	•	Fine-Grained Pruned Model:
	•	Accuracy: ~91%
	•	Model Size: 8.2 MB
	•	Inference Time: 2.3 ms
	•	Channel Pruned Model:
	•	Accuracy: ~90%
	•	Model Size: 6.5 MB
	•	Inference Time: 1.7 ms

References

	•	CIFAR-10 Dataset
	•	VGG Network
	•	PyTorch Pruning Documentation
