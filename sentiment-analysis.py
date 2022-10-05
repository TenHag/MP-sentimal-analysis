import pandas as pd

df =pd.read_csv('/content/IMDB Dataset.csv')
df.head()
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()
docs = np.array(['We are a group of students studying in AI&DS, Dhruv, Ashraf, Aryan and Saud'])

bag = vect.fit_transform(docs)
print(vect.vocabulary_)
print(bag.toarray())

from sklearn.feature_extraction.text import TfidfTransformer
np.set_printoptions(precision = 2)
tfidf = TfidfTransformer(use_idf = True, norm = '', smooth_idf = True)
print(tfidf.fit_transform(bag).toarray())

import nltk
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(
    use_idf = True,
    norm = '',
    smooth_idf = True
)
