# Stock Market Direction Predictor- LSTM classifier
A deep learning project that predicts the next-day price direction (UP/DOWN) of RELIANCE.NS (Reliance Industries, NSE) using an LSTM neural network trained on 10 years of daily price data.

**Final Accuracy:** 54.88% ,**BUY Accuracy:** 53.96%, 
**SELL Accuracy:** 55.70%. Signal Split is balanced (BUY: 202, SELL:228)

## What it does
Downloads 10 years of daily OHLCV (Open, High, Low, Close, Volume) data via yfinance, engineers 9 technical indicators as features, then trains a single-layer LSTM classifier on 60-day rolling windows to predict next-day price direction. Output is a BUY or SELL signal with a predicted probability.
## Features used
Return , MA7, MA21, Volatility, Volume_change, RSI, MACD, MACD_Histogram, Bollinger Band Width- all scaled with MinMaxScaler before training.
## Model architecture
```
LSTM(32) -> Dropout(0.3) -> Dense(16, relu) -> Dense(1, sigmoid)
```

Adam optimizer (lr=0.0005), Binary crossentropy loss. Class weights applied to handle imbalanced BUY/SELL distribution and prevent the model from defaulting to one signal.

## Results
-Overall directional accuracy : **54.88%** (50%= random guessing)\
-Accuracy when signal is BUY: 53.96%\
-Accuracy when signal is SELL: 55.70%\
-Signal distribution: BUY 202, SELL 228 -reasonably balanced\
-Predicted probability range: 0.481 - 0.516 (std dev: 0.0078)
## Debugging and what I ruled out
The 54.88% is a genuine result. I spent a significant time verifying this before accepting it.


**Bug I found and fixed:** The sequence builder was passing train labels into test evalution, giving the wrong results. Fixed this before drawing any conclusion.

**Things I tested to rule out as causes of low accuracy:**\
-Early Stopping cutting training short- Tested patience 5, 10, 15, 20. No improvement observed.\
-Insufficient model capacity- Tried sracked LSTM (128 -> 64). Model performed worse than single LSTM(32)\
-Model collapse- Checked prediction standard deviation (0.0078) to confirm model is not predicting near-constant values.

The tight probability range (all predictions between 0.48–0.52) shows the model is genuinely uncertain about daily price movement. 54–55% directional accuracy is consistent with published literature on LSTM stock direction prediction using daily data.

**Cause of low accuracy:**\
Stock prices don't depend on price history alone. Real market movement is driven by factors like global news, economic events, company announcements, and investor sentiment — none of which are captured in OHLCV data. This is a fundamental limitation of any purely technical indicator-based model.
## Stack
Python 3.10, TensorFlow 2.10, Keras, yfinance, scikit-learn, pandas, matplotlib.\
Trained locally on RTX 4050 (CUDA 11.2 / cuDNN 8.1)
## Files
-`stock_direction_predictor_lstm_model.ipynb`- Fulll pipeline from data download to evalution\
-`best_lstm_classifier.h5`-saved model weights (best val_accuracy checkpoint)
## How to run
```bash
pip install tensorflow==2.10 yfinance scikit-learn padas matplotlib
```

Open the notebook in Jupyter and run all cells. Data doenloads live via yfinance - no local dataset needed.
## Requirements

- Python 3.10
- TensorFlow 2.10
- Keras
- yfinance
- scikit-learn
- pandas
- numpy
- matplotlib
