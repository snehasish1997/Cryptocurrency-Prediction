import requests
import pandas as pd
import datetime as dt
import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Reshape, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.layers import LayerNormalization, Dropout, Add
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from env_var import env
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

## Start of custom Transformer model implementation with CNN


def evaluate_model_performance_mape(y_test, y_pred):
    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # MAPE as a percentage
    
    # Calculate accuracy based on MAPE
    accuracy = 100 - mape

    # Output results
    print(f"MAPE: {mape}%")
    print(f"Accuracy: {accuracy}%")

    return accuracy

def transformer_encoder_model(inputs, head_size, no_of_head, feed_forward, dropout=0):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=no_of_head, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = Add()([x, inputs])  # Residual connection

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(feed_forward, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return Add()([x, res])

def cnn_transformer_hybrid_model(input_shape, head_size, no_of_heads, feed_forward, no_of_transformer_blocks, mlp, cnn_filters, kernel_size, dropout=0, mlp_dropout=0):
    inputs = Input(shape=input_shape)

    # CNN layers to extract features
    x = Conv1D(filters=cnn_filters, kernel_size=kernel_size, activation="relu", padding="same")(inputs)
    x = Conv1D(filters=cnn_filters, kernel_size=kernel_size, activation="relu", padding="same")(x)
    # Note: No Flatten here; keep the sequence structure intact
    x = LayerNormalization()(x)

    # Transformer layers
    for _ in range(no_of_transformer_blocks):
        x = transformer_encoder_model(x, head_size, no_of_heads, feed_forward, dropout)

    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(1)(x)
    return Model(inputs, outputs)

## End of custom Transformer model implementation with CNN



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



def scaling_main_data(df, features, time_step):
    scaler = MinMaxScaler()  # Use MinMaxScaler to handle a large range of data better
    scaled_data = scaler.fit_transform(df[features])
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
    # Check if the test size is valid
    if test_size == 0 or len(X) == 0:
        raise ValueError("Test size is too small or dataset is empty.") 
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]  
    return X_train, X_test, y_train, y_test


def train_model(model, X_train, y_train, epochs=40, batch_size=4, validation_split=0.2):
    # Adding EarlyStopping and ReduceLROnPlateau callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    Reduced_LPR = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)    
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split,
              callbacks=[early_stopping, Reduced_LPR])
    return model

def save_model(model, scaler, model_filename='cnn_hybrid_model.h5', scaler_filename='scaler.save'):
    model.save(model_filename)
    joblib.dump(scaler, scaler_filename)
    print("Model and scaler saved successfully.")

    
    
def main():
    # Fetch data
    end_date = dt.datetime.now().strftime('%Y-%m-%d')

    df = fetch_cryptocurrency_data('BTC', 'USD', '2021-01-01', end_date, 'api_key', 2000)
    
    df.to_csv('data.csv')
    
    # Prepare data
    features = env.features
    time_step = env.lag
    X, y, scaler = scaling_main_data(df, features, time_step)
    
    test_size = 0.25
    X_train, X_test, y_train, y_test = split_data(X, y, test_size)

    
    # Build and train CNN-Transformer model
    input_shape = (time_step, len(features))
    cnn_hybrid_model = cnn_transformer_hybrid_model(
        input_shape=input_shape,
        head_size=128,  # Reduced model complexity
        no_of_heads=4,
        feed_forward=128,  # Reduced model complexity
        no_of_transformer_blocks=3,  # Reduced model complexity
        mlp = [64],  # Reduced model complexity
        cnn_filters=32,  # Reduced model complexity
        kernel_size=3,
        dropout=0.2,  # Increased dropout
        mlp_dropout=0.2  # Increased dropout
    )
    
    cnn_hybrid_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    cnn_hybrid_model = train_model(cnn_hybrid_model, X_train, y_train, epochs=40, batch_size=4, validation_split=0.2)
    
    y_pred = cnn_hybrid_model.predict(X_test)
    
    y_test_unscaled = scaler.inverse_transform(np.concatenate((X_test[:, -1, :-1], y_test.reshape(-1, 1)), axis=1))[:, -1]
    y_pred_unscaled = scaler.inverse_transform(np.concatenate((X_test[:, -1, :-1], y_pred), axis=1))[:, -1]
    
    accuracy = evaluate_model_performance_mape(y_test_unscaled, y_pred_unscaled)
    
    
    print(accuracy)
    
    # Save model and scaler
    save_model(cnn_hybrid_model, scaler)
    
    # Load model and scaler (if needed)
    model = load_model('cnn_transformer_model.h5')
    scaler = joblib.load('scaler.save')
    
    
    

if __name__ == '__main__':
    main()
