import pandas as pd
df=pd.read_csv("/Users/anishjain/Downloads/netflix_data.csv")
df.head()
print(df.tail())  # ending 5 rows
print(df.isnull().sum())
print(df.size)    # size
print(df.shape)   # no of rows and columns

print(df.describe())  #  describe the columns 
print(df.columns)
df.fillna('', inplace=True)
print(df.isnull().sum())
# merge two columns
df['tags']=df['type']+df['description']
print(df.head())
# making a new dataset
new_df=df[['show_id','title','type','description','tags']]
print(new_df.head())
# droping of two columns
new_df=new_df.drop(columns=['type','description'])
new_df.head()

# countVectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=10000,stop_words='english')
print(cv)

vec=cv.fit_transform(new_df['tags'].values.astype('U')).toarray()
print(vec)

from sklearn.metrics.pairwise import cosine_similarity
sim=cosine_similarity(vec)
print(sim)

new_df[new_df['title']=='Dick Johnson Is Dead']

dist=sorted(list(enumerate(sim[0])),reverse=True,key=lambda vec:vec[1])
print(dist)
for i in dist[0:12]:
    print(new_df.iloc[i[0]].title)

def recommend(movies):
    index=new_df[new_df['title']==movies].index[0]
    distance=sorted(list(enumerate(sim[index])),reverse=True,key=lambda vec:vec[1])
    for i in distance[0:5]:
        print(new_df.iloc[i[0]].title)
recommend("Kota Factory")      
      
      
