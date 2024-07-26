import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import  tree

from sklearn.preprocessing import LabelEncoder # type: ignore
df=pd.read_csv("C:\\Users\\acer\\Downloads\\titanic.csv")
print(df.head(5))


median_age=math.floor(df['Age'].mean())
print(median_age)
df['Age'] = df['Age'].fillna(median_age)
print(df.head(10))
df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
print(df.head(10))

inputs=df.drop('Survived',axis="columns")
target=df["Survived"]

print(inputs.head(5))
print(target.head(5))

le_Sex=LabelEncoder()
inputs['Sex_n']=le_Sex.fit_transform(inputs['Sex'])
inputs_n=inputs.drop(['Sex'] ,axis='columns')

print(inputs_n.head(10))
inputs_n.Age[:10]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs_n,target,test_size=0.2)
print(len(X_train))
print(len(X_test))
print(X_test)

# train our model 
model=tree.DecisionTreeClassifier()
model.fit(X_train,y_train)

# predicting a score
print(model.score(X_test,y_test))
