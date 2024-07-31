from sklearn.cluster import KMeans
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
#%matplotlib inline
from sklearn.datasets import load_digits
digits=load_digits()
print(dir(digits))
print(digits.target)
print(digits.target_names)
df=pd.DataFrame(digits.data,digits.target)
print(df.head(5))

df['target']=digits.target
print(df.head(20))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('target',axis='columns'), df.target, test_size=0.3)

print(len(X_train))
print(len(X_test))
from sklearn.neighbors import KNeighborsClassifier
#knn = KNeighborsClassifier(n_neighbors=3)
knn = KNeighborsClassifier(n_neighbors=13)

knn.fit(X_train,y_train)
print(knn.score(X_test,y_test))

from sklearn.metrics import confusion_matrix
y_pred=knn.predict(X_test)
#print(y_pred)
cn=confusion_matrix(y_test,y_pred)
print(cn)

import seaborn as sn
plt.figure(figsize=(9,4))
sn.heatmap(cn,annot=True)
plt.xlabel('predicted')
plt.ylabel('Truth')
plt.show()


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
