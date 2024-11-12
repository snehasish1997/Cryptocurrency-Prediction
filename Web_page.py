import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.dates as mdates
from env_var import env
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import RobustScaler

# Streamlit app title
st.title('Cryptocurrency Price Prediction')

# Input parameters in the sidebar
st.sidebar.header('Input Parameters')

st.sidebar.text('This is all the information that you are predicting')

crypto_symbol = st.sidebar.text_input('Cryptocurrency Symbol', 'BTC')
comparison_symbol = st.sidebar.text_input('Comparison Currency', 'USD')
start_date = st.sidebar.date_input('Start Date', dt.date(2021, 1, 1))
end_date = st.sidebar.date_input('End Date', dt.date.today())
time_step = st.sidebar.number_input('Time Step (Lag)', min_value=1, max_value=100, value=100, step=1)
forecast_horizon = 12



# Main function
def main():
    scaler = joblib.load('scaler.save')
    model = load_model('lstm_model.h5')
    features = env.features  
    
    if st.button('Fetch and Predict'):
        #st.info('Fetching Data...')
        
        df = pd.read_csv('data.csv')
        
        
        if not df.empty:
            st.success('Data fetched successfully!')
            st.subheader('Sample of Raw Data')
            st.write(df.head())
            
            # Ensure enough data for prediction
            if len(df) <= time_step:
                st.error(f'Not enough data points. At least {time_step + 1} data points are required.')
                return
            
            st.info('Predicting prices...')
            
            df_scaled = scaler.transform(df[features])
    
            X = []
            for i in range(time_step, len(df_scaled)):
                X.append(df_scaled[i-time_step:i])
    
            X = np.array(X)
    
            # Ensure correct input shape for LSTM (samples, time steps, features)
            X = X.reshape(X.shape[0], X.shape[1], len(features))
    
            predicted_scaled = model.predict(X)
            predicted_scaled = np.squeeze(predicted_scaled, axis=-1)
    
            # Prepare array for inverse transformation
            inverse_predicted = np.zeros((predicted_scaled.shape[0], len(features)))
            inverse_predicted[:, features.index('close')] = predicted_scaled
    
            # Inverse transform to get actual prices
            predicted_prices = scaler.inverse_transform(inverse_predicted)[:, features.index('close')]
            
            
            # Predict future prices for the next 12 hours
            last_sequence = df_scaled[-time_step:]  # Last sequence to start predictions
            future_predictions = []
            for _ in range(forecast_horizon):
                last_sequence = last_sequence.reshape((1, time_step, len(features)))
                future_pred_scaled = model.predict(last_sequence)
                future_pred_scaled = np.squeeze(future_pred_scaled, axis=-1)
                
                # Prepare for inverse transform
                inverse_future = np.zeros((1, len(features)))
                inverse_future[:, features.index('close')] = future_pred_scaled
                
                # Inverse transform to get actual price
                future_predicted_price = scaler.inverse_transform(inverse_future)[:, features.index('close')][0]
                future_predictions.append(future_predicted_price)
                
                # Update the last sequence with the predicted value
                new_sequence = np.append(last_sequence[:, 1:, :], [inverse_future], axis=1)
                last_sequence = new_sequence
                
            # Assign predicted prices to the dataframe
            df['Predicted'] = np.nan
            df.iloc[time_step:, df.columns.get_loc('Predicted')] = predicted_prices
            
            #st.subheader('Actual vs Predicted Prices lawra')
            #st.write(df[['close', 'Predicted']].tail())
            
            
            # Debugging: print some of the predicted prices
            #st.write("Sample of predicted prices:", predicted_prices[:10])
            
            # Align predicted prices with the corresponding dates
            df_pred = df.iloc[time_step:].copy()
            df_pred['Predicted'] = predicted_prices
            
            #st.subheader('Actual vs Predicted Prices')
            #st.write(df_pred[['close', 'Predicted']].tail())
            
            # Plot actual vs predicted prices
            df_pred['time'] = pd.to_datetime(df_pred['time'])
            df_pred = df_pred.sort_values('time')
            df_pred.set_index('time', inplace=True)

            
            plt.figure(figsize=(14, 7))
            plt.plot(df_pred.index, df_pred['close'], label='Actual Price', color='blue')
            plt.plot(df_pred.index, df_pred['Predicted'], label='Predicted Price', color='red')

            plt.xlabel('Date')
            plt.ylabel(f'Price ({comparison_symbol})')
            plt.title(f'{crypto_symbol}/{comparison_symbol} Price Prediction')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)
            
            import time
            import datetime
            current_time = time.time()
            current_time_int = int(current_time)
            
            
            
            
            
            # Display the next 12 predicted prices
            st.subheader('Next 12 Hour Predictions')
            for i, price in enumerate(future_predictions):
                timestamp = current_time+1
                dt_object = datetime.datetime.fromtimestamp(timestamp)
                st.write(f"Hour {dt_object}: {price:.2f} {comparison_symbol}")
            
            
        else:
            st.error('No data fetched for the given parameters. Please check your input and try again.')

if __name__ == '__main__':
    main()
