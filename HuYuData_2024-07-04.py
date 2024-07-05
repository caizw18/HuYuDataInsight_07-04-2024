import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

# Generate dummy sequential data
def generate_data(seq_length, num_samples):
    X = np.random.rand(num_samples, seq_length, 1)
    y = np.sum(X, axis=1)
    return X, y

seq_length = 10
num_samples = 1000
X, y = generate_data(seq_length, num_samples)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(seq_length, 1), activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Make predictions
y_pred = model.predict(X_test)
print(f'Predictions: {y_pred[:5]}')
print(f'Actual: {y_test[:5]}')