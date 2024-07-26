from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
#%matplotlib inline
from sklearn.datasets import load_iris
iris=load_iris()

df = pd.DataFrame(iris.data,columns=iris.feature_names)
print(df.head(5))

df['flower']=iris.target
print(df.head(5))

df.drop(['sepal length (cm)', 'sepal width (cm)', 'flower'],axis='columns',inplace=True)
print(df.head(3))

plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.scatter(df['petal length (cm)'], df['petal width (cm)'])
plt.show()

Km =KMeans(n_clusters=3)
print(Km)

y_predicted=Km.fit_predict(df[['petal length (cm)','petal width (cm)']])
print(y_predicted)

df['cluster']=y_predicted
print(df)

df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]

plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color='green')
plt.scatter(df2['petal length (cm)'],df2['petal width (cm)'],color='red')
plt.scatter(df3['petal length (cm)'],df3['petal width (cm)'],color='yellow')
plt.show()


sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['petal length (cm)','petal width (cm)']])
    sse.append(km.inertia_)
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)
plt.show()
