import pandas as pd
data = pd.read_csv(r"/Users/anishjain/Downloads/amazon_alexa.tsv", delimiter = '\t', quoting = 3)
print(data.head())
print(data.tail())  # ending 5 rows
print(data.size)    # size
print(data.shape)   # no of rows and columns

print(data.describe())  #  describe the columns 
print(data.columns)
print(data.isnull().sum()) # checking a null values 
data[data['verified_reviews'].isna() == True]
data.dropna(inplace=True)
print(data.shape)
data['length'] = data['verified_reviews'].apply(len)
print(data.head())
print(data.dtypes)

#Distinct values of 'rating' and its count  
print(f"Rating value count: \n{data['rating'].value_counts()}")
import matplotlib.pyplot as plt
data['rating'].value_counts().plot.bar(color = 'red')
plt.title('Rating distribution count')
plt.xlabel('Ratings')
plt.ylabel('Count')
plt.show()

#Distinct values of 'feedback' and its count 
print(f"Feedback value count: \n{data['feedback'].value_counts()}")
#Extracting the 'verified_reviews' value for one record with feedback = 0

review_0 = data[data['feedback'] == 0].iloc[1]['verified_reviews']
print(review_0)

review_1=data[data['feedback']==1].iloc[1]['verified_reviews']
print(review_1)

data['feedback'].value_counts().plot.bar(color = 'blue')
plt.title('Feedback distribution count')
plt.xlabel('Feedback')
plt.ylabel('Count')
plt.show()

#Feedback = 0
data[data['feedback'] == 0]['rating'].value_counts()
#Feedback = 1
data[data['feedback'] == 1]['rating'].value_counts()
#Distinct values of 'variation' and its count 

print(f"Variation value count: \n{data['variation'].value_counts()}")

data['variation'].value_counts().plot.bar(color='grey')
plt.title("Variation distribution count")
plt.xlabel('Variation')
plt.ylabel("Count")
plt.show()

print(data['length'].describe())

data.groupby('variation')['rating'].mean()
data.groupby('variation')['rating'].mean().sort_values().plot.bar(color='brown',figsize=(11,6))
plt.title(" Mean rating according to variation")
plt.xlabel("Variation")
plt.ylabel("Mean rating")
plt.show()

import seaborn as sns
sns.histplot(data['length'],color='blue').set(title='Distribution of length of review ')

sns.histplot(data[data['feedback']==0]['length'],color='red').set(title="Distribution of length of review if feedback 0")

sns.histplot(data[data['feedback']==1]['length'],color='red').set(title="Distribution of length of review if feedback 1")

data.groupby('length')['rating'].mean().plot.hist(color="blue",figsize=(7,6),bins=20)
plt.title(" Review length wise mean ratings")
plt.xlabel('ratings')
plt.ylabel('length')
plt.show()

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(stop_words='english')
words=cv.fit_transform(data.verified_reviews)

print(data.head())


reviews=" ".join([review for review in data ['verified_reviews']])
from wordcloud import WordCloud
wc=WordCloud(background_color='white',max_words=50)

plt.figure(figsize=(10,10))
plt.imshow(wc.generate(reviews))
plt.title('WordCloud for all reviews',fontsize=10)
plt.axis('off')
plt.show

neg_reviews=" ".join([review for review in data[data['feedback']==0]['verified_reviews']])
neg_reviews=neg_reviews.lower().split()

pos_reviews=" ".join([review for review in data[data['feedback']==1]['verified_reviews']])
pos_reviews = pos_reviews.lower().split()


unique_negative = (x for x in neg_reviews if x not in pos_reviews)
unique_negative=" ".join(unique_negative)

unique_positive=[x for x in pos_reviews if x not in neg_reviews]
unique_positive=" ".join(unique_positive)

wc=WordCloud(background_color='white',max_words=50)

plt.figure(figsize=(10,10))
plt.imshow(wc.generate(unique_negative))
plt.title('WordCloud for negative reviews',fontsize=10)
plt.axis('off')
plt.show


wc=WordCloud(background_color='white',max_words=50)

plt.figure(figsize=(10,10))
plt.imshow(wc.generate(unique_positive))
plt.title('WordCloud for positive reviews',fontsize=10)
plt.axis('off')
plt.show

import nltk
nltk.download('stopwords',halt_on_error=False)
import re
from nltk.stem.porter import PorterStemmer 
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
     
corpus = []
stemmer = PorterStemmer()
for i in range(0, data.shape[0]):
  review = re.sub('[^a-zA-Z]', ' ', data.iloc[i]['verified_reviews'])
  review = review.lower().split()
  review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
  review = ' '.join(review)
  corpus.append(review)

cv = CountVectorizer(max_features = 2500)

#Storing independent and dependent variables in X and y
X = cv.fit_transform(corpus).toarray()
y = data['feedback'].values
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 15)

print(f"X train: {X_train.shape}")
print(f"y train: {y_train.shape}")
print(f"X test: {X_test.shape}")
print(f"y test: {y_test.shape}")

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier


#Accuracy of the model on training and testing data
scaler = MinMaxScaler()
X_train_scl = scaler.fit_transform(X_train)
X_test_scl = scaler.transform(X_test)
model_rf = RandomForestClassifier()
model_rf.fit(X_train_scl, y_train)
RandomForestClassifier()
print("Training Accuracy :", model_rf.score(X_train_scl, y_train))
print("Testing Accuracy :", model_rf.score(X_test_scl, y_test))


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_preds = model_rf.predict(X_test_scl)
print(y_preds)
#Confusion Matrix
cm = confusion_matrix(y_test, y_preds)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model_rf.classes_)
cm_display.plot()
plt.show()

