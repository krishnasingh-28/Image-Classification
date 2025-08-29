# Image Classification with a Convolutional Neural Network (CNN)

## Project Overview

This project demonstrates how to build and train a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset. The goal is to create a model that can accurately classify images into one of ten categories.

## Dependencies

The following libraries are required to run this project:

* **TensorFlow:** An open-source machine learning framework.

* **Keras:** A high-level neural networks API, running on top of TensorFlow.

* **NumPy:** A library for the Python programming language, adding support for large, multi-dimensional arrays and matrices.

* **Matplotlib:** A plotting library for the Python programming language and its numerical mathematics extension NumPy.

## Dataset: CIFAR-10

The CIFAR-10 dataset is a collection of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

The 10 classes are:
`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`

## Methodology

### 1. Data Loading and Preparation

* **Load Data:** The CIFAR-10 dataset is loaded directly from `tensorflow.keras.datasets`.

* **Normalization:** The pixel values of the images are normalized to be between 0 and 1 by dividing by 255. This helps the neural network train more efficiently.

* **One-Hot Encoding:** The integer labels are converted to one-hot encoded vectors. For example, a label of `3` becomes `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`.

### 2. Model Architecture

A sequential CNN model is built with the following layers:

1.  **First Convolutional Block:**

    * `Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))`: This layer has 32 filters of size 3x3. It detects basic features like edges and corners.

    * `MaxPooling2D((2, 2))`: This layer reduces the spatial dimensions of the output from the convolutional layer.

2.  **Second Convolutional Block:**

    * `Conv2D(64, (3, 3), activation='relu')`: This layer has 64 filters and learns more complex patterns.

    * `MaxPooling2D((2, 2))`: Further reduces the dimensions.

3.  **Classification Layers:**

    * `Flatten()`: Converts the 2D feature maps into a 1D vector.

    * `Dense(64, activation='relu')`: A fully connected layer with 64 neurons.

    * `Dense(10, activation='softmax')`: The output layer with 10 neurons (one for each class). The softmax activation function outputs a probability distribution over the classes.

### 3. Compilation and Training

* **Compilation:** The model is compiled with the `adam` optimizer, `categorical_crossentropy` as the loss function, and `accuracy` as the evaluation metric.

* **Training:** The model is trained for 10 epochs with a batch size of 64.

## Evaluation

After training, the model is evaluated on the test set.

**Test Accuracy:** `0.6980`

## Making Predictions

The model can be used to make predictions on individual images. Here is an example of a prediction on an image from the test set:

**Predicted class:** cat
**True class:** cat

## Conclusion

This project successfully implements a CNN for image classification on the CIFAR-10 dataset. The model achieves a reasonable accuracy and can be used as a baseline for further improvements, such as adding more layers, using data augmentation, or trying different architectures.