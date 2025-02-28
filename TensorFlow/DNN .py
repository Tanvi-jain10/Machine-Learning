import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

# Load dataset
df = pd.read_csv("/Users/anishjain/Downloads/train.csv")

# Select features and target
features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex"]
target = "Survived"

# Drop missing values
df_cleaned = df[features + [target]].dropna()

# Encode 'Sex' (male=1, female=0)
df_cleaned["Sex"] = LabelEncoder().fit_transform(df_cleaned["Sex"])

# Split data into features (X) and target (Y)
X = df_cleaned[features].values
Y = df_cleaned[target].values.reshape(-1, 1)  # Reshape for TensorFlow compatibility

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Normalize features (helps with training stability)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build Deep Neural Network (DNN) Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),  # First hidden layer
    tf.keras.layers.Dense(32, activation="relu"),  # Second hidden layer
    tf.keras.layers.Dense(16, activation="relu"),  # Third hidden layer
    tf.keras.layers.Dense(1, activation="sigmoid")  # Output layer (binary classification)
])

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(X_train, Y_train, epochs=100, batch_size=16, verbose=1, validation_data=(X_test, Y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=1)
print(f"Final Test Accuracy: {test_acc * 100:.2f}%")
