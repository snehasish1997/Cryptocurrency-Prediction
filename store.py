duplicates = df['time'].duplicated().sum()
print(f'Number of duplicate timestamps: {duplicates}')



df.set_index('time', inplace=True)
print(df.head())  # Check the first few rows to ensure data alignment



plt.figure(figsize=(14, 7))
plt.plot(df['time'], df['close'], label='Actual Price', color='blue')
plt.plot(df['time'], df['Predicted'], label='Predicted Price', color='red')
plt.xlabel('Date')
plt.ylabel(f'Price ({comparison_symbol})')
plt.title(f'{crypto_symbol}/{comparison_symbol} Price Prediction')
plt.legend()
plt.grid(True)
st.pyplot(plt)



print(f"Length of 'close': {len(df['close'])}")
print(f"Length of 'Predicted': {len(df['Predicted'].dropna())}")



print(df[['close', 'Predicted']].head(20))  # Visualize the first 20 rows


data.isna().sum()
test = test.drop(columns=['ranknow'])
final_data = final_data.drop(columns=['ranknow'])

test3_data = test2_data[(test2_data['volume'] > 0) & (test2_data['symbol'] == 'BTC')]