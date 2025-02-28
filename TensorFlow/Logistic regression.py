import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
train_df=pd.read_csv("/Users/anishjain/Downloads/train.csv")
print(train_df.head())

print(train_df.describe())
print(train_df.columns)
print(train_df.shape)




# Select features and target
features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex"]
target = "Survived"

# Drop missing values
df_cleaned = train_df[features + [target]].dropna()

# Encode 'Sex' (male=1, female=0)
df_cleaned["Sex"] = LabelEncoder().fit_transform(df_cleaned["Sex"])

# Split data into features (X) and target (Y)
X = df_cleaned[features].values
Y = df_cleaned[target].values.reshape(-1, 1)

# Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build a logistic regression model using a single neuron with sigmoid activation
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(X_train.shape[1],), activation="sigmoid")
])

# Compile model with binary cross-entropy loss
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(X_train, Y_train, epochs=200, verbose=1, validation_data=(X_test, Y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=1)
print(f"Final Test Accuracy: {test_acc * 100:.2f}%")
 

