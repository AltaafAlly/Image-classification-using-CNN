# CNN Image Classification

This project demonstrates the implementation of a Convolutional Neural Network (CNN) for binary image classification. The code trains a CNN model using different loss functions and learning rates to compare their performance.

## Dataset

The dataset is structured as follows:
- Training data: `train/train_data`
- Testing data: `test/test_data`

The images are preprocessed and fed into the model using data generators from Keras.

## Model Architecture

The CNN model consists of the following layers:
- Convolutional layers with ReLU activation
- Max pooling layers
- Flatten layer
- Dense layers with ReLU activation and dropout regularization
- Output layer with sigmoid activation

The model is compiled with different loss functions and optimizers to evaluate their impact on performance.

## Training and Evaluation

The code iterates over different loss functions (`binary_crossentropy`, `hinge`, `mean_squared_error`) and trains the model for each loss function. The training progress and validation accuracy are plotted for comparison.

Additionally, the code explores the effect of different learning rates (`0.001`, `0.01`, `0.1`) on the model's performance. The test accuracy for each learning rate is plotted in a bar chart.

## Overfitting Mitigation

To address overfitting, dropout layers are added to the model. The code trains the model with the `binary_crossentropy` loss function and a learning rate of `0.001`, incorporating dropout regularization.

## Confusion Matrix

The code generates a confusion matrix to visualize the model's performance using the `binary_crossentropy` loss function. The confusion matrix is plotted as a heatmap using Seaborn.

## Dependencies

The following libraries are used in this project:
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

Make sure to have these libraries installed before running the code.

## Usage

1. Ensure that the dataset is placed in the correct directories (`train/train_data` and `test/test_data`).
2. Run the code in a Python environment with the required dependencies installed.
3. The code will train the model with different loss functions and learning rates, and display the resulting plots and confusion matrix.

Feel free to modify the code and experiment with different model architectures, hyperparameters, and evaluation techniques to further improve the classification performance.
