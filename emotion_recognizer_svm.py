#important libraries
import pandas as pd
import numpy as np
import nltk
import re
#importing stopwords is optional, in this case it decreased accuracy
#from nltk.corpus import stopwords
import itertools

import os
os.chdir('/home/Akai/Downloads')


data = pd.read_csv('text_emotion.csv')
#data = data.iloc[:50,:]
unique = data.sentiment.unique()

uniqueness = list()
for i in range(len(unique)):
    uniqueness.append(unique[i])

#converting categories to ordered and numbered categories  
data.sentiment = data.sentiment.astype("category", ordered=True, categories=uniqueness).cat.codes

#stopset = set(stopwords.words('english'))

from nltk.stem.wordnet import WordNetLemmatizer 
lem = WordNetLemmatizer()

#removal of urls
for i in range(len(data)):
    data.content[i] = re.sub(r"http\S+", "", data.content[i])

#removal of html phrases
for i in range(len(data)):
    data.content[i] = data.content[i].split()
    index = 0
    for j in range(len(data.content[i])):
        if data.content[i][j][0] == '@':
            index = j
    data.content[i] = np.delete(data.content[i], index)
    words = data.content[i][0]
    for k in range(len(data.content[i])-1):
        words+= " " + data.content[i][k+1]
    data.content[i] = words
    
#comprehensive cleaning           

for i in range(len(data)):
    data.content[i] = re.sub(r'[^\w]', ' ', data.content[i])
    data.content[i] = ''.join(''.join(s)[:2] for _, s in itertools.groupby(data.content[i]))
    data.content[i] = data.content[i].replace("'", "")
    data.content[i] = nltk.tokenize.word_tokenize(data.content[i])
    #data.content[i] = [w for w in data.content[i] if not w in stopset]
    for j in range(len(data.content[i])):
        data.content[i][j] = lem.lemmatize(data.content[i][j], "v")
    if len(data.content[i]) == 0:
        data = data.drop(data.index[[i]])

        
data = data.reset_index(drop=True)
for i in range(len(data)):
    words = data.content[i][0]
    for j in range(len(data.content[i])-1):
        words+= ' ' + data.content[i][j+1]
    data.content[i] = words
        
        
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data.content, data.sentiment, test_size=0.25, random_state=0)

x_train = x_train.reset_index(drop = True)
x_test = x_test.reset_index(drop = True)

y_train = y_train.reset_index(drop = True)
y_test = y_test.reset_index(drop = True)

vectorizer = TfidfVectorizer(min_df=4, max_df=0.9)

train_vectors = vectorizer.fit_transform(x_train)
test_vectors = vectorizer.transform(x_test)

model = svm.SVC(kernel='linear') 
model.fit(train_vectors, y_train) 
predicted_sentiment = model.predict(test_vectors)

print(classification_report(y_test, predicted_sentiment))

predicted_sentiments = []
for s in range(len(predicted_sentiment)):
    predicted_sentiments.append(uniqueness[predicted_sentiment[s]])

for z in range(len(y_train)):
    y_train[z] = uniqueness[y_train[z]]
    
prediction_df = pd.DataFrame({'Content':x_test, 'Emotion_predicted':predicted_sentiment, 'Emotion_actual': y_train})
prediction_df.to_csv('emotion_recognizer.csv', index = False)
