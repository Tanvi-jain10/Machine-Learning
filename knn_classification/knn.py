import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
#%matplotlib inline
from sklearn.datasets import load_iris
iris=load_iris()
print(dir(iris))
df = pd.DataFrame(iris.data,columns=iris.feature_names)
print(iris.feature_names)
print(df.head(5))
df['target']=iris.target
print(df.head(5))

# different types of flower
print(iris.target_names)

print(df[df.target==0].head(6))
print(df[df.target==1].head(6))
print(df[df.target==2].head(6))





df['flower_names']=df.target.apply(lambda x: iris.target_names[x])
print(df)              

df0=df[:50]
df1=df[50:100]
df2=df[100:]

plt.scatter(df0['sepal length (cm)'] ,df0['sepal width (cm)'],color='red',marker='+')
plt.scatter(df1['sepal length (cm)'] ,df1['sepal width (cm)'],color='green',marker='.')
#plt.scatter(df2['sepal length (cm)'] ,df2['sepal width (cm)'],color='blue',marker='+')
plt.xlabel('length')
plt.ylabel('width')
plt.show()

plt.scatter(df0['petal length (cm)'] ,df0['petal width (cm)'],color='red',marker='+')
plt.scatter(df1['petal length (cm)'] ,df1['petal width (cm)'],color='green',marker='.')
#plt.scatter(df2['petal length (cm)'] ,df2['petal width (cm)'],color='blue',marker='+')
plt.xlabel('length')
plt.ylabel('width')
plt.show()

X=df.drop(['target','flower_names'],axis='columns')
print(X.head(4))
y=df.target
print(y.head(5))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

print(len(X_train))
print(len(X_test))

from sklearn.neighbors import KNeighborsClassifier
#knn = KNeighborsClassifier(n_neighbors=3)
knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(X_train,y_train)
print(knn.score(X_test,y_test))

from sklearn.metrics import confusion_matrix
y_pred=knn.predict(X_test)
#print(y_pred)
cn=confusion_matrix(y_test,y_pred)
print(cn)

import seaborn as sn
plt.figure(figsize=(7,3))
sn.heatmap(cn,annot=True)
plt.xlabel('predicted')
plt.ylabel('Truth')
plt.show()


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
