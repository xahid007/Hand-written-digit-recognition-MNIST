# Hand-Written-Digit-Recognition-MNIST

This repository contains code for a handwritten digit recognition project using neural networks. The goal of the project is to train a model that can accurately classify handwritten digits from the MNIST dataset.

## MNIST Dataset

The MNIST dataset is a widely used benchmark dataset in the field of machine learning and computer vision. It consists of a collection of grayscale images of handwritten digits ranging from 0 to 9, making it ideal for digit recognition tasks. The dataset has been modified from the original dataset provided by the National Institute of Standards and Technology (NIST).

### Dataset Details

- Image Samples: The dataset comprises 70,000 grayscale images, each with a resolution of 28x28 pixels. This results in a total of 784 pixels per image, represented as 2D arrays or flattened into 1D vectors for analysis.

- Training and Test Sets: The MNIST dataset is divided into a training set and a test set. The training set contains 60,000 images, while the test set contains 10,000 images. This split allows for training models on a subset of the data and evaluating their performance on unseen samples.

- Label Information: Each image is associated with a label indicating the corresponding digit it represents (ranging from 0 to 9). These labels provide ground truth information for supervised learning tasks, where the objective is to predict the correct digit based on the input image.

### Application in Machine Learning

The MNIST dataset serves as a standard benchmark for evaluating the performance of machine learning algorithms, particularly in the domain of image classification and digit recognition. Researchers and practitioners often utilize the dataset as a starting point to develop and test their models, making it a widely accepted reference for comparing and assessing different techniques.

### Accessibility

The MNIST dataset is publicly available and easily accessible from various machine learning libraries and repositories. It is commonly included as a built-in dataset in popular deep learning frameworks such as TensorFlow and PyTorch, simplifying the process of loading and working with the data.

**Note:** While the MNIST dataset is a valuable resource for learning and experimentation, it is essential to recognize that its simplicity and distinct characteristics may not fully represent real-world challenges in computer vision tasks.




## Model Architecture

The neural network model used for this project is a simple sequential model implemented with TensorFlow's Keras API. The model architecture consists of the following layers:

- Flatten layer: Converts the 2D input images into a 1D array.
- Dense layer (ReLU activation): Hidden layer with 100 units and Rectified Linear Unit (ReLU) activation function.
- Dense layer (Sigmoid activation): Output layer with 10 units (one for each digit) and Sigmoid activation function.

## Training and Evaluation

The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as the evaluation metric. It is trained on the training dataset for 10 epochs.

To evaluate the trained model, it is tested on the separate test dataset, and the accuracy and loss values are calculated using the `model.evaluate()` function.

## Usage

1. Install the required dependencies specified in the `requirements.txt` file.
2. Run the `ipynb` script to train the model on the MNIST dataset.

Feel free to explore and modify the code to experiment with different architectures, hyperparameters, and evaluation metrics.

## Contributing

Contributions to this project are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

