# LSTM-based Stock Market Prediction for Tesla Inc.

This repository contains a Jupyter Notebook that demonstrates how to predict Tesla stock prices using an LSTM (Long Short-Term Memory) neural network. The dataset is sourced from Yahoo Finance and feature engineering techniques are applied to improve the model's performance.

## Dataset

The stock market data for Tesla Inc. (TSLA) is fetched from Yahoo Finance using the `yfinance` library. The dataset contains historical daily stock prices including Open, High, Low, Close, Adjusted Close, and Volume.

## Requirements

To run the notebook, you need the following libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- yfinance

## Model Architecture

The model is built using the TensorFlow library and consists of the following layers:

1. LSTM layer with 50 units and `relu` activation function
2. Dropout layer with a dropout rate of 0.2
3. Dense layer with a single output neuron

The model is compiled with the Adam optimizer and Mean Squared Error (MSE) loss function.

## Model Performance

The performance of the model is evaluated using the Root Mean Squared Error (RMSE) metric on both the training and test sets. In this example, the model achieved an RMSE of approximately 1.49 on the training set and 14.59 on the test set. The model's performance may vary depending on the selected features, window size, and other hyperparameters.

## Visualization of Predictions

The graph below shows the actual Tesla stock prices (in blue) along with the predicted prices from the LSTM model for both the training (in green) and test (in red) sets. Although the model's predictions do not perfectly align with the actual price curve, they generally follow the same overall trend. For instance, when the stock price is decreasing, the model's predictions also tend to decrease, though not to the same extent as the actual price. This level of performance may be suitable for long-term investment strategies, where the focus is on capturing the general market trend rather than precise price movements. However, for short-term trading strategies such as intraday trading, the model's current performance may not be adequate, as it does not accurately predict the exact fluctuations in stock prices.


![Tesla Stock Price Predictions](Tesla%20Stock%20Price%20Predictions.png)


## Model Performance

The performance of the model is evaluated using the Root Mean Squared Error (RMSE) metric on both the training and test sets. In this example, the model achieved an RMSE of approximately 1.49 on the training set and 14.59 on the test set. The model's performance may vary depending on the selected features, window size, and other hyperparameters.

## Disclaimer

Please note that the performance of the model on real-world data may be different. The stock market is influenced by various factors, and predicting stock prices accurately is a challenging task due to the randomness and various factors influencing the stock market. The train and test accuracies may vary significantly because of this randomness. This project serves as a demonstration of the LSTM model's capabilities and should not be used for making actual investment decisions. Make sure to thoroughly validate your model and use caution when making decisions based on its output.
