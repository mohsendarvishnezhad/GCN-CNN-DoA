import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Simulation parameters
num_antennas = 12
num_samples = 5000
angle_range = (-60, 60)  # degrees
snr_db = 13  # Signal-to-Noise Ratio in dB
carrier_freq = 100e6  # Hz (not directly used, assume normalized wavelength=1)
d = 0.5  # spacing in terms of wavelength

# Convert SNR from dB to linear scale
snr_linear = 10 ** (snr_db / 10)

# Generate a simple adjacency matrix for ULA:
# Each antenna is connected to its immediate neighbors (and itself)
A = np.eye(num_antennas)
for i in range(num_antennas):
    if i > 0:
        A[i, i-1] = 1.0
    if i < num_antennas - 1:
        A[i, i+1] = 1.0
A = tf.constant(A, dtype=tf.float32)

def steering_vector(angle_deg, num_antennas):
    """
    Compute the steering vector for a given angle (in degrees) for a ULA.
    The vector is defined as: a = exp(1j * π * n * sin(θ))
    where n = 0, 1, ..., num_antennas-1 and θ is in radians.
    """
    angle_rad = np.deg2rad(angle_deg)
    n = np.arange(num_antennas)
    a = np.exp(1j * np.pi * n * np.sin(angle_rad))
    return a

def add_noise(signal, snr_linear):
    """
    Add complex Gaussian noise to a signal based on the specified SNR.
    """
    power_signal = np.mean(np.abs(signal) ** 2)
    noise_power = power_signal / snr_linear
    noise = np.sqrt(noise_power/2) * (np.random.randn(*signal.shape) + 1j*np.random.randn(*signal.shape))
    return signal + noise

# Data generation: For simplicity, we assume a single source per sample.
# The input feature will be the real and imaginary parts of the received signal (shape: (num_antennas, 2))
X = []
y = []  # true angles in degrees
for _ in range(num_samples):
    angle = np.random.uniform(angle_range[0], angle_range[1])
    a = steering_vector(angle, num_antennas)
    # Simulate received signal at each antenna (use an amplitude of 1)
    received = a
    received_noisy = add_noise(received, snr_linear)
    # Separate real and imaginary parts
    features = np.stack((np.real(received_noisy), np.imag(received_noisy)), axis=-1)  # shape (num_antennas, 2)
    X.append(features)
    y.append(angle)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

print("Data shape:", X.shape, "Labels shape:", y.shape)

# Split data into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a custom GCN layer
class SimpleGCN(layers.Layer):
    def __init__(self, units, activation=tf.nn.relu, **kwargs):
        super(SimpleGCN, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        
    def build(self, input_shape):
        # input_shape: (batch, num_nodes, features)
        self.weight = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(SimpleGCN, self).build(input_shape)
        
    def call(self, inputs, adjacency):
        # Graph convolution: A * X * W
        x = tf.matmul(inputs, self.weight)
        x = tf.matmul(adjacency, x)
        return self.activation(x)

# Build the hybrid CNN-GCN model
def build_model(num_antennas, num_features):
    input_layer = layers.Input(shape=(num_antennas, num_features))  # (batch, 12, 2)
    
    # CNN branch: Use 1D convolutions along the antenna axis
    cnn_branch = layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_layer)
    cnn_branch = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(cnn_branch)
    cnn_branch = layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(cnn_branch)
    cnn_branch = layers.Flatten()(cnn_branch)
    
    # GCN branch: Use our custom GCN layers
    # Note: We pass the fixed adjacency matrix as a constant to the GCN layers.
    gcn_branch = SimpleGCN(units=128)(input_layer, adjacency=A)
    gcn_branch = SimpleGCN(units=64)(gcn_branch, adjacency=A)
    gcn_branch = layers.Flatten()(gcn_branch)
    
    # Combine features from both branches
    combined = layers.Concatenate()([cnn_branch, gcn_branch])
    
    # Prediction block (MLP)
    dense = layers.Dense(256, activation='relu')(combined)
    dense = layers.Dense(128, activation='relu')(dense)
    dense = layers.Dense(64, activation='relu')(dense)
    output = layers.Dense(1, activation='linear')(dense)
    
    model = models.Model(inputs=input_layer, outputs=output)
    return model

model = build_model(num_antennas, 2)
model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
model.summary()

# Train the model
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stop])

# Evaluate the model
loss, mae = model.evaluate(X_val, y_val)
print("Validation MAE:", mae)

# Plot training history
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Training History')
plt.show()

# Make predictions on validation set and compute RMSE
y_pred = model.predict(X_val).flatten()
rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
print("Validation RMSE: {:.2f}°".format(rmse))

# Plot true vs predicted angles for a subset of validation samples
plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_pred, alpha=0.6)
plt.xlabel('True Angle (°)')
plt.ylabel('Predicted Angle (°)')
plt.title('True vs. Predicted DOA')
plt.plot([angle_range[0], angle_range[1]], [angle_range[0], angle_range[1]], 'r--')
plt.show()
