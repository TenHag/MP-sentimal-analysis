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

y = df.sentiment.values
x = tfidf.fit_transform(df['review'].values.astype('U'))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1,test_size = 0.5, shuffle = 'False')

'''LogisticRegression,
LogisticRegressionCV,'''
import pickle 
from sklearn.linear_model import LogisticRegressionCV
clf = LogisticRegressionCV(cv = 5,
                           scoring = 'accuracy',
                           random_state = 0,
                           n_jobs = -1,
                           verbose = 3,
                           max_iter = 300).fit(x_train, y_train)
                           
saved_model = open('saved model.sav','wb')
pickle.dump(clf, saved_model)
saved_model.close()

filename = 'saved_model.sav'
saved_clf = pickle.load(open(filename, 'rb'))

saved_clf.score(x_test, y_test)

test = ["SIUUUUUUUUU"]
X_test=tfidf.transform(test)
saved_clf.predict(X_test)

