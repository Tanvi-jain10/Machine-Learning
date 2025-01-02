import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC
 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor 
file_path="/Users/anishjain/Downloads/Large_Dataset_With_Null_Values.csv"
df=pd.read_csv(file_path)
print(df.head())  # starting 5 rows 
print(df.tail())  # ending 5 rows
print(df.size)    # size
print(df.shape)   # no of rows and columns

print(df.describe())  #  describe the columns 
print(df.columns)     # give columns name
 
# check null values 


print(df.isnull().sum())   

df2=df.dropna(subset=['Gender'])  # drop the rows which have null values in gender columns

print(df2['Gender'].isnull().sum())  
# Label Encoding
# giving label to male and female as 0 & 1

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df2['Gender'] = label_encoder.fit_transform(df2['Gender'].astype(str))


# checking the values of gender 0,1
unique_values = df2['Gender'].unique()
print(unique_values)

unique_values = df2['Customer ID'].unique()
print(unique_values)

print(df2.shape)


#  filling null values in name column
df2['Name'] = df2.apply(
    lambda row: f"Customer_{row['Customer ID']}" if pd.isnull(row['Name']) else row['Name'], axis=1
)
unique_values = df2['Name'].unique()
print(unique_values)


print(df2['Name'].isnull().sum()) 


# filling null values in age column 
df2['Age'].fillna(df2['Age'].mean(),inplace=True)

print(df2['Age'].isnull().sum()) 

unique_values = df2['Age'].unique()
print(unique_values)

# filling null values in Annual Income (k$)
df2['Annual Income (k$)'].fillna(df2['Annual Income (k$)'].mean(),inplace=True)

print(df2['Annual Income (k$)'].isnull().sum()) 

unique_values = df2['Annual Income (k$)'].unique()
print(unique_values)

# filling null values in Spending Score (1-100)
df2['Spending Score (1-100)'].fillna(df2['Spending Score (1-100)'].mean(),inplace=True)

print(df2['Spending Score (1-100)'].isnull().sum()) 

unique_values = df2['Spending Score (1-100)'].unique()
print(unique_values)


# outliers 
import matplotlib.pyplot as plt

plt.boxplot(df2['Annual Income (k$)'].dropna())
plt.title('Boxplot of Annual Income')
plt.ylabel('Annual Income (k$)')
plt.show()

plt.boxplot(df2['Spending Score (1-100)'].dropna())
plt.title('Boxplot of Spending Score ')
plt.ylabel('Spending Score (1-100)')
plt.show()

# Use scatter plots to identify outliers visually in a feature against another

plt.scatter(df2['Age'], df2['Annual Income (k$)'])
plt.title('Scatter Plot of Age vs Annual Income')
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.show()


plt.scatter(df2['Spending Score (1-100)'], df2['Annual Income (k$)'])
plt.title('Scatter Plot of Age vs Annual Income')
plt.xlabel('Spending Score (1-100)')
plt.ylabel('Annual Income (k$)')
plt.show()



# first we have  apply logistic regression for predictin gender based on another feratures 
X = df2[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]  # Feature columns
y = df2['Gender']  # Target column (Gender)

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression model
log_reg = LogisticRegression()

# Train the model
log_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = log_reg.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# applying SVM 
svm_model = SVC(kernel='linear')  # You can experiment with different kernels (linear, rbf, poly)

# Train the model
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# to predict the spending score  using svm 

# Select features (excluding 'Spending Score (1-100)' which is the target)
X = df2[['Age', 'Annual Income (k$)']]  # Feature columns
y = df2['Spending Score (1-100)']  # Target column (Spending Score)

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling for Random Forest (scaling is not mandatory but can sometimes help)
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Create and train the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  # 100 trees in the forest
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model's performance using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")



# Optirfonally, print some predicted vs actual values for inspection
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison.head())

file_path = "/Users/anishjain/Downloads/Updated_File.csv"  # Path to save the updated file

# Save the updated DataFrame to a new CSV file in the Downloads folder
df2.to_csv(file_path, index=False)

print(f"File saved to: {file_path}")
