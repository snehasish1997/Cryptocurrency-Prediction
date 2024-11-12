from sklearn.metrics import mean_absolute_percentage_error
import requests
import pandas as pd
import datetime as dt
import numpy as np
import joblib
import datetime as dt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from env_var import env
import matplotlib.pyplot as plt
import requests
from sklearn.preprocessing import RobustScaler


def evaluate_model_performance_mape(y_test, y_pred):
    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # MAPE as a percentage
    
    # Calculate accuracy based on MAPE
    accuracy = 100 - mape

    # Output results
    print(f"MAPE: {mape}%")
    print(f"Accuracy: {accuracy}%")

    return accuracy

def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.DataFrame(gain, index=df.index).rolling(window=period).mean()
    avg_loss = pd.DataFrame(loss, index=df.index).rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)  
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = rsi
    return df


def fetch_cryptocurrency_data(crypto_name, crypto_currency, start_date, end_date, api_key, limit):
    base_url = 'https://min-api.cryptocompare.com/data/v2/histohour'  
    # Convert dates to timestamps
    endTime = int(dt.datetime.strptime(end_date, '%Y-%m-%d').timestamp())
    startTime = int(dt.datetime.strptime(start_date, '%Y-%m-%d').timestamp())   
    my_data = []  
    while endTime > startTime:
        url = f'{base_url}?fsym={crypto_name}&tsym={crypto_currency}&limit={limit}&toTs={endTime}&api_key={api_key}'   
        response = requests.get(url)
        data = response.json()   
        if 'Data' not in data or 'Data' not in data['Data']:
            break   
        df = pd.DataFrame(data['Data']['Data']) 
        if df.empty:
            break
        my_data.append(df)
        # Update toTs to the time of the oldest fetched data point
        endTime = df['time'].min() - 1
    # Combine all dataframes
    if my_data:
        final_df = pd.concat(my_data, ignore_index=True)
        final_df['time'] = pd.to_datetime(final_df['time'], unit='s')
        final_df = final_df[(final_df['time'] >= pd.to_datetime(start_date)) & 
                          (final_df['time'] <= pd.to_datetime(end_date))]
        final_df.set_index('time', inplace=True)
        #final_df = calculate_rsi(final_df)
        return final_df
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no data was fetched



def scaling_main_date(df, features, time_step):
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(df[features])
    #return scaler, scaled_data
    joblib.dump(scaler, 'scaler.save')
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], len(features))
    return X, y, scaler
    
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i])
        y.append(data[i, -1])  # target is 'close'
    return np.array(X), np.array(y)


def split_data(X, y, test_ratio):
    test_size = int(len(X) * test_ratio)
    X_train, X_test = X[: -test_size], X[-test_size:]
    y_train, y_test = y[: -test_size], y[-test_size:]
    return X_train, X_test, y_train, y_test

def build_model(time_step, feature_count):
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=(time_step, feature_count)))
    model.add(LSTM(units=64, return_sequences=True))
    model.add(LSTM(units=32, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_model(model, X_train, y_train, epochs=40, batch_size=4, validation_split=0.2):
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
    return model

def save_model(model, scaler, model_filename='lstm_model.h5', scaler_filename='scaler.save'):
    model.save(model_filename)
    joblib.dump(scaler, scaler_filename)
    print("Model and scaler saved successfully.")
    



def main():
    # Fetch data
    end_date = dt.datetime.now().strftime('%Y-%m-%d')

    df = fetch_cryptocurrency_data('BTC', 'USD', '2021-01-01', end_date, 'your_api_key_here', 2000)

    
    df.to_csv('data.csv')
    
    # Prepare data
    features = env.features
    time_step = env.lag
    X, y, scaler = scaling_main_date(df, features, time_step)
    
    test_size = 0.25
    X_train, X_test, y_train, y_test = split_data(X, y, test_size)

    
    # Build and train model
    model = build_model(time_step, len(features))
    
    
    
    model = train_model(model, X_train, y_train, epochs=40, batch_size=4, validation_split=0.2)
    
    y_pred = model.predict(X_test)

    # Inverse scaling the predicted and test data (if scaling was applied)
    y_test_unscaled = scaler.inverse_transform(np.concatenate((X_test[:, -1, :-1], y_test.reshape(-1, 1)), axis=1))[:, -1]
    y_pred_unscaled = scaler.inverse_transform(np.concatenate((X_test[:, -1, :-1], y_pred), axis=1))[:, -1]
    
    accuracy = evaluate_model_performance_mape(y_test_unscaled, y_pred_unscaled)
    
    
    print(accuracy)

    
    # Save model and scaler
    save_model(model, scaler)

if __name__ == '__main__':
    main()


    