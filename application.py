import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# App Title
st.title('📈 Stock Price Forecasting using LSTM')

# User Input for Stock Symbol
stock_symbol = st.text_input('Enter Stock Symbol (Example: AAPL for Apple)', 'AAPL')

if stock_symbol:
    # Fetch Data from Yahoo Finance
    st.info(f'Fetching data for {stock_symbol}...')
    data = yf.download(stock_symbol, period='5y')

    if not data.empty:
        st.success('Data fetched successfully!')
        data = data[['Close']]

        st.subheader(f'{stock_symbol} Stock Data (Last 5 Years)')
        st.write(data.tail())

        # Plot Historical Data
        st.subheader('Stock Price Trend')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data['Close'], label='Close Price')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        ax.legend()
        st.pyplot(fig)

        # Prepare Data
        dataset = data.values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        training_data_len = int(np.ceil(len(scaled_data) * 0.8))
        train_data = scaled_data[0:training_data_len, :]

        x_train = []
        y_train = []

        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Build LSTM Model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        with st.spinner('Training LSTM model...'):
            model.fit(x_train, y_train, batch_size=32, epochs=10)

        # Prepare Test Data
        test_data = scaled_data[training_data_len - 60:, :]
        x_test = []
        y_test = dataset[training_data_len:, :]

        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # Predictions
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        st.write(f'Root Mean Squared Error: {rmse:.2f}')

        # Visualize Validation Predictions
        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid['Predictions'] = predictions

        st.subheader('Forecasted vs Actual')
        fig2, ax2 = plt.subplots(figsize=(14, 7))
        ax2.plot(train['Close'], label='Training Data')
        ax2.plot(valid[['Close', 'Predictions']], label='Validation and Predictions')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Price (USD)')
        ax2.legend()
        st.pyplot(fig2)

        # Future Forecasting
        st.subheader('🔮 Forecast Future Stock Prices')
        forecast_days = st.slider('Select how many days to forecast', min_value=1, max_value=60, value=30)

        last_60_days = scaled_data[-60:]
        future_input = last_60_days.reshape((1, 60, 1))

        future_predictions = []

        for _ in range(forecast_days):
            future_pred = model.predict(future_input)
            future_predictions.append(future_pred[0, 0])
            future_input = np.concatenate((future_input[:, 1:, :], future_pred.reshape(1, 1, 1)), axis=1)

        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        future_dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='B')

        # Plot Future Predictions
        st.subheader('📅 Future Stock Price Forecast')
        fig3, ax3 = plt.subplots(figsize=(14, 7))
        ax3.plot(data['Close'], label='Historical Data')
        ax3.plot(future_dates, future_predictions, label='Future Forecast', color='red')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Price (USD)')
        ax3.legend()
        st.pyplot(fig3)

        st.success('Forecasting Completed!')

    else:
        st.error('Failed to fetch data. Please check the stock symbol and try again.')