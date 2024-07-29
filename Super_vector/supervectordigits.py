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
from sklearn.svm import SVC
rbf_model = SVC(kernel='rbf')

print(len(X_train))
print(len(X_test))

rbf_model.fit(X_train, y_train)
print(rbf_model.score(X_test,y_test))

linear_model = SVC(kernel='linear')
linear_model.fit(X_train,y_train)
print(linear_model.score(X_test,y_test))
