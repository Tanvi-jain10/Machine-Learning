import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# Load the data
df =pd.read_csv("C:\\Users\\acer\\Downloads\\insurance_data.csv")
df.head()
plt.scatter(df.age,df.bought_insurance, marker='+',color='red')

print(df)
x_train,x_test,y_train,y_test = train_test_split(df[['age']],df.bought_insurance,test_size=0.1)
print(x_test)
print(y_test)

model=LogisticRegression()
model.fit(x_test,y_test)
predication=model.predict(x_test) 
print(predication)
print(model.score(x_test,y_test))
#print(model.predict(65))
#print(model.predict.proba(x_test))
