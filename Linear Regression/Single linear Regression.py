import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as linear_model

# Load the data
df = pd.read_csv("C:\\Users\\acer\\Desktop\\homeprices.csv")
print(df)

# Plot the data
plt.xlabel('area (sqr ft)')
plt.ylabel('price (US $)')
plt.scatter(df.area, df.price, color='red', marker='+')
plt.show()

# Create and train the linear regression model
reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)

# Predict the price for an area of 3300 sq ft
predicted_price = reg.predict([[5444]])
print(predicted_price)
value=reg.coef_
print(value)
print(reg.intercept_)

