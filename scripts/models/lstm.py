import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error

X = np.load("data/processed/model_ready/X.npy")
y = np.load("data/processed/model_ready/y.npy")

split = int(len(X)*0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential([
    LSTM(32, input_shape=(X.shape[1], 1)),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X_train[...,None], y_train, epochs=10)

pred = model.predict(X_test[...,None])
mae = mean_absolute_error(y_test, pred)
print("LSTM MAE:", mae)
