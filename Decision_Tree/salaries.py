import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import  tree

from sklearn.preprocessing import LabelEncoder # type: ignore
df=pd.read_csv("C:\\Users\\acer\\Downloads\\salaries.csv")
print(df.head(5))

#creting a new table without target columns

inputs=df.drop('salary_more_then_100k',axis="columns")
target=df["salary_more_then_100k"]

print(inputs.head(5))
print(target.head(5))

# convert label columns to number
le_company=LabelEncoder()
le_job=LabelEncoder()
le_degree=LabelEncoder()

# inserting in new columns

inputs['company_n']=le_company.fit_transform(inputs['company'])
inputs['job_n']=le_company.fit_transform(inputs['job'])
inputs['degree_n']=le_company.fit_transform(inputs['degree'])

print(inputs.head(5))

# droping label columns
inputs_n=inputs.drop(['company','job','degree'],axis='columns')
print(inputs_n.head(5))

# train our model 
model=tree.DecisionTreeClassifier()
model.fit(inputs_n,target)

# predicting a score
print(model.score(inputs_n,target))

print(model.predict([[2,2,1]]))
