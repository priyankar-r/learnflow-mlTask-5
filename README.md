# learnflow-mlTask-5
## Netflix Stock Price Prediction
### Overview
This repository contains code for a simple stock price prediction model using a neural network implemented in TensorFlow/Keras. The model predicts the stock's low price and is trained on the Netflix Stock Price dataset sourced from Kaggle.

## Dataset
The dataset used for this project is the Netflix Stock Price Prediction dataset from Kaggle. It includes historical stock prices for Netflix, covering the period from 2005-01-01 to 2017-12-31.

link for the dataset :https://www.kaggle.com/datasets/jainilcoder/netflix-stock-price-prediction

## Loading and Preprocessing Data:
Loads the dataset 'NFLX.csv' using Pandas and preprocesses it for model training. Converts non-numeric columns, fills missing values, and creates target classes based on quantiles. 

## Model Training:
Utilizes TensorFlow's Keras API to build and train a neural network model for stock price prediction. Compiles the model with appropriate loss and optimizer functions. Trains the model on the prepared dataset and evaluates its accuracy

## Dependencies
Make sure you have the following Python libraries installed:
pandas
numpy
scikit-learn
matplotlib
seaborn
tensorflow
warnings
