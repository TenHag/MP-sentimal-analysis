import pandas as pd

df =pd.read_csv('/content/IMDB Dataset.csv')
df.head()
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()
docs = np.array(['I am Dhruv, studying in AIDS'
                 'Please like,share,comment and Subscribe to my channel'
                 'thanks for all the support to my channel'])

bag = vect.fit_transform(docs)
print(vect.vocabulary_)
print(bag.toarray())
