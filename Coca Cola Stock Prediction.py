# Live Updating System

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

st.title("📈 Coca-Cola Stock Price Predictor")

# Load data
df = yf.download('KO', start='2010-01-01')
st.write(df.tail())

# Preprocess
scaler = MinMaxScaler()
data = df[['Close']].values
scaled_data = scaler.fit_transform(data)
test_data = scaled_data[-(60+1):]

X_test = []
X_test.append(test_data[:60, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Load model
model = load_model('model.h5')
prediction = model.predict(X_test)
predicted_price = scaler.inverse_transform(prediction)

st.subheader("Predicted Next Day Closing Price:")
st.metric(label="Predicted Price ($)", value=round(predicted_price[0][0], 2))
