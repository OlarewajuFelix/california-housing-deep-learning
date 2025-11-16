California Housing Price Prediction with Deep Learning
Overview

This project demonstrates house price prediction using deep learning on the California Housing dataset.
The goal is to teach regression modeling with tabular data, including normalization, training a neural network, and evaluating performance with MAE/MSE.

Features

Regression task: predicts continuous house prices (in hundreds of thousands of dollars).

Input features are standardized using StandardScaler.

Neural network with two hidden layers (64 neurons each, ReLU activation).

Evaluation using Mean Absolute Error (MAE) and Mean Squared Error (MSE).

Optional scatter plot showing predicted vs actual prices.

Dataset

Source: sklearn.datasets.fetch_california_housing

Features (8 columns):

MedInc — median income in block group

HouseAge — median house age in block group

AveRooms — average rooms per household

AveBedrms — average bedrooms per household

Population — block group population

AveOccup — average household occupancy

Latitude — block group latitude

Longitude — block group longitude

Target: Median house value for California districts (in hundreds of thousands)

Requirements

Python 3.10+ (tested on 3.13)

Packages:

pip install numpy scikit-learn tensorflow matplotlib

How to Run

Clone or download this repository.

Open terminal / PowerShell in the project folder.

Run the script:

python house_price_regression.py


Output will include:

Training progress per epoch

Test MAE

Predictions vs actual values for the first 5 test samples

Optional scatter plot (requires matplotlib)

Example Output
Training the model...
Epoch 1/100 ...
...
Epoch 100/100 ...
Test MAE: 0.52

Predictions for first 5 test samples: [0.76 1.23 0.85 0.95 1.10]
Actual values for first 5 test samples: [0.81 1.30 0.90 0.95 1.15]

Notes

MAE (Mean Absolute Error): average difference between predicted and actual prices.

Normalization: scaling features improves model convergence and stability.

Verbose training: shows progress; you can set verbose=0 to hide it.

Optional plot: helps visualize how well predictions match actual prices.