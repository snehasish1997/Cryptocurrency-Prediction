import streamlit as st
import pandas as pd
import numpy as np
import joblib
from env_var import env
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


#from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Input, Dense, LSTM, Concatenate
#from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
#from sklearn.preprocessing import RobustScaler
#from tensorflow.keras.layers import MultiHeadAttention, Layer


#import matplotlib.pyplot as plt
import datetime as dt
import requests
from sklearn.preprocessing import RobustScaler

# Streamlit app title
st.title('Cryptocurrency Price Prediction')

# Input parameters in the sidebar
st.sidebar.header('Input Parameters')

st.sidebar.text('dont change anything')

crypto_symbol = st.sidebar.text_input('Cryptocurrency Symbol', 'BTC')
comparison_symbol = st.sidebar.text_input('Comparison Currency', 'USD')
start_date = st.sidebar.date_input('Start Date', dt.date(2019, 1, 1))
end_date = st.sidebar.date_input('End Date', dt.date.today())
time_step = st.sidebar.number_input('Time Step (Lag)', min_value=1, max_value=100, value=100, step=1)


# Main function
def main():
    scaler = joblib.load('scaler.save')
    model = load_model('lstm_model.h5')
    features = env.features  
    #df=df.load(filename='data.csv')
    ###new things
    if st.button('Fetch and Predict'):
        st.info('Fetching data...')
        #df = fetch_crypto_data(crypto_symbol, comparison_symbol, start_date, end_date, api_key, 2000)
        
        #df = df.to_csv('data.csv')
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
            
            
            ##start from here
            
            
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
     
            #predicted_prices = predict_price(df, features, time_step)
            
            # Debugging length of predicted prices
            st.write(f'Length of predicted prices: {len(predicted_prices)}')
            st.write(f'Expected length: {len(df) - time_step}')
            
            # Assign predicted prices to the dataframe
            df['Predicted'] = np.nan
            df.iloc[time_step:, df.columns.get_loc('Predicted')] = predicted_prices
            
            st.subheader('Actual vs Predicted Prices')
            st.write(df[['close', 'Predicted']].head())
            
            # Check for NaN values in predicted prices
            if df['Predicted'].isna().sum() > 0:
                st.error(f"There are {df['Predicted'].isna().sum()} NaN values in the Predicted column.")
                
            
            #ch=df.copy()
            #ch['time'] = pd.to_datetime(ch['time'])
            #ch.set_index('time', inplace=True)

            #ch.index = ch.index.date
            #print(df.head())
            #df['time'] = pd.to_datetime(df['time'])
            #df.set_index('time', inplace=True)
            
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time')  # Ensure the time column is sorted
            print(df['time'].head())
            df.set_index('time', inplace=True)
            
            #df.index = df.index.date
            #print("start last")
            #print(df.tail())
            
            
            #df['time'] = pd.to_datetime(df['time'])  # Convert 'time' to datetime if not already
            #df['date'] = df['time'].dt.date  # Create a new column 'date' with only the date part

            # Optionally, set 'time' as the index
            #df.set_index('time', inplace=True)
            #print(df.head())

            # Plot actual vs predicted prices
            plt.figure(figsize=(14, 7))
            plt.plot(df.index, df['close'], label='Actual Price', color='blue')
            plt.plot(df.index, df['Predicted'], label='Predicted Price', color='red')
            plt.xlabel('Date')
            plt.ylabel(f'Price ({comparison_symbol})')
            plt.title(f'{crypto_symbol}/{comparison_symbol} Price Prediction')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)
        else:
            st.error('No data fetched for the given parameters. Please check your input and try again.')

if __name__ == '__main__':
    main()
