import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.models import load_model
from sklearn.metrics import mean_squared_error


# Load the dataset
file_path = 'apple_stock_data.csv'
df = pd.read_csv(file_path)

#Display the first few rows of the dataframe
print(df.head())
print("-------------------------------------------------------------------------------------------------------------")
#Display information about the dataset
print(df.info())
print("-------------------------------------------------------------------------------------------------------------")
#Get summary statistics
print(df.describe())
print("-------------------------------------------------------------------------------------------------------------")

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort by date
df = df.sort_values('Date')

# Set 'Date' as the index
df.set_index('Date', inplace=True)

# Select all the columns
data = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]

# Check if any value is missing in the entire DataFrame
print(data.isnull().values.any())
print("-------------------------------------------------------------------------------------------------------------")
# Check for missing values in each column
print(data.isnull().sum())


# Calculate the correlation matrix
correlation_matrix = data.corr()

# Print the correlation matrix
print(correlation_matrix)

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# Select the relevant columns
data = df[['Open', 'High', 'Low', 'Close', 'Volume']]

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)


# Create a DataFrame with the scaled data
scaled_df = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
# Display the first few rows of the scaled dataframe
print(scaled_df.head())
# Display the statistics of the 'Close' column to verify normalization
close_stats = scaled_df['Close'].describe()
print(close_stats)
# Check if the minimum value is 0 and the maximum value is 1
print(f"Min value: {scaled_df['Close'].min()}")
print(f"Max value: {scaled_df['Close'].max()}")

# Split the data into training, validation, and test sets
train_size = int(len(scaled_df) * 0.7)
val_size = int(len(scaled_df) * 0.15)
train_data = scaled_df[:train_size]
val_data = scaled_df[train_size:train_size + val_size]
test_data = scaled_df[train_size + val_size:]

# Display the shapes of the datasets
print(f"Training data shape: {train_data.shape}")
print(f"Validation data shape: {val_data.shape}")
print(f"Test data shape: {test_data.shape}")

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length][3]  # Target is the 'Close' price
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 60  # You can adjust this value
X_train, y_train = create_sequences(train_data.values, seq_length)
X_val, y_val = create_sequences(val_data.values, seq_length)
X_test, y_test = create_sequences(test_data.values, seq_length)

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Validation data shape: {X_val.shape}, {y_val.shape}")
print(f"Test data shape: {X_test.shape}, {y_test.shape}")


# Define the LSTM model
model = keras.Sequential()
model.add(keras.layers.LSTM(units=50, return_sequences=True, input_shape=(seq_length, 5)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(units=50, return_sequences=False))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

# Save the model
model.save('apple_stock_lstm_model.keras')



# Load the trained model

model = load_model('apple_stock_lstm_model.keras')

# Make predictions on the test set
predicted_prices = model.predict(X_test)

# Inverse transform the predictions and the actual values to their original scale
predicted_prices = scaler.inverse_transform(np.concatenate([np.zeros((predicted_prices.shape[0], 4)), predicted_prices], axis=1))[:, -1]
actual_prices = scaler.inverse_transform(np.concatenate([np.zeros((y_test.shape[0], 4)), y_test.reshape(-1, 1)], axis=1))[:, -1]

# Calculate the mean squared error
mse = mean_squared_error(actual_prices, predicted_prices)
print(f"Mean Squared Error on Test Set: {mse}")

# Plot the predicted vs actual prices
plt.figure(figsize=(14, 5))
plt.plot(actual_prices, color='blue', label='Actual Apple Stock Price')
plt.plot(predicted_prices, color='red', label='Predicted Apple Stock Price')
plt.title('Apple Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Apple Stock Price')
plt.legend()
plt.show()