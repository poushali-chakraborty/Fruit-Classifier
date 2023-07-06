# Fruit Classifier

This project demonstrates how to train a fruit classifier using convolutional neural networks (CNN) on the Quick, Draw! dataset. The code is implemented in a Jupyter Notebook (`doodle classifier.ipynb`) and utilizes TensorFlow and Keras libraries.

## Dataset

The Quick, Draw! dataset is a collection of doodles from millions of users around the world. For this project, we focus on four fruits: apple, banana, grapes, and pineapple. The dataset contains numpy arrays (`.npy` files) of 28x28 grayscale images representing the doodles of each fruit.

To access the dataset, the notebook includes commands to download the necessary files from the Google Cloud Storage.

## Code Overview

The Jupyter Notebook (`doodle classifier.ipynb`) is organized into the following sections:

1. **Data Preparation**: This section loads the fruit doodle images from the downloaded `.npy` files. It reshapes the images to 28x28x1 and prepares the labels for training.

2. **Model Architecture**: The notebook defines a CNN model using Keras. The model consists of two convolutional layers, max pooling, dropout regularization, and fully connected layers.

3. **Model Compilation**: The model is compiled with the categorical cross-entropy loss function, Adam optimizer, and accuracy metric.

4. **Model Training**: The model is trained on the training data using the `model.fit()` function. The training is done for a specified number of epochs, and the training progress, including loss and accuracy, is recorded.

5. **Model Evaluation**: After training, the model is evaluated on the test data by making predictions using the `model.predict()` function. The predicted probabilities are converted to class labels, and the accuracy of the model is calculated.

6. **Training Progress Visualization**: The loss and accuracy values during training are plotted using `matplotlib.pyplot` to visualize the model's training progress.

7. **Prediction**: An example doodle image from the test set is displayed, and the model predicts its fruit class. The predicted and actual classes are printed.

8. **Model Saving**: The trained model is saved in the HDF5 format as `fruits.h5` for future use.

## How to Use

To run the code in the Jupyter Notebook, follow these steps:

1. Install the required dependencies: TensorFlow, Keras, NumPy, Matplotlib, scikit-learn, and Pillow.

2. Download the Jupyter Notebook (`doodle classifier.ipynb`) and open it in a Jupyter Notebook environment.

3. Run each cell in the notebook sequentially to execute the code step-by-step.

4. Observe the training progress, model accuracy, and the prediction for an example doodle image.

You can modify the code to add more fruits or experiment with different model architectures, hyperparameters, and optimization algorithms to improve the classifier's performance.



## Acknowledgments

The Quick, Draw! dataset used in this project is publicly available and maintained by Google. The code in this repository is inspired by the TensorFlow and Keras documentation and examples.

## References

1. [Quick, Draw! Dataset](https://quickdraw.withgoogle.com/data)
2. [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
3. [Keras Documentation](https://keras.io/api/)
4. [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
5. [Pillow Documentation](https://pillow.readthedocs.io/en/stable/)
