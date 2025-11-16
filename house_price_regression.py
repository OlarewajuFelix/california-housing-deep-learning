# House Price Prediction - Regression (California Housing)
# Pylance-friendly version using tf.keras

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Optional: Import matplotlib for plotting if installed
try:
    import matplotlib.pyplot as plt
    plotting_available = True
except ImportError:
    plotting_available = False

# 1️⃣ Load dataset
california = fetch_california_housing()
X = california.data       # Features
y = california.target     # Target: house prices in hundreds of thousands

# 2️⃣ Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3️⃣ Normalize / scale input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4️⃣ Build the model using tf.keras
model = tf.keras.Sequential([
    tf.keras.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# 5️⃣ Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 6️⃣ Train the model with verbose=1 to show progress
print("Training the model...")
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=1)

# 7️⃣ Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f"\nTest MAE: {mae:.2f}")

# 8️⃣ Make predictions for first 5 samples
predictions = model.predict(X_test[:5])
print("\nPredictions for first 5 test samples:", predictions.flatten())
print("Actual values for first 5 test samples:", y_test[:5])

# 9️⃣ Optional: Scatter plot
if plotting_available:
    pred_all = model.predict(X_test).flatten()
    plt.scatter(y_test, pred_all, alpha=0.5)
    plt.xlabel("Actual Prices (hundreds of thousands)")
    plt.ylabel("Predicted Prices")
    plt.title("Predicted vs Actual House Prices")
    plt.show()
else:
    print("\nMatplotlib not installed; skipping plot.")
