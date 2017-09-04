#important libraries
import pandas as pd
import numpy as np
import nltk
import re
#importing stopwords is optional, in this case it decreased accuracy
#from nltk.corpus import stopwords
import itertools
import time


start_time = time.time()

import os
os.chdir('/home/ankushraut/Downloads')


data = pd.read_csv('text_emotion.csv')
#data = data.iloc[:100,:]


#stopset = set(stopwords.words('english'))

from nltk.stem.wordnet import WordNetLemmatizer 
lem = WordNetLemmatizer()

#comprehensive cleaning
for i in range(len(data)):
    data.content[i] = re.sub(r"http\S+", "", data.content[i])
    if len(data.content[i]) == 0:
        data.drop(i, inplace = True)
    else:
        data.content[i] = data.content[i].split()
        index = 0
        for j in range(len(data.content[i])):
            if data.content[i][j][0] == '@':
                index = j
        data.content[i] = np.delete(data.content[i], index)
        if len(data.content[i]) == 0:
            data.drop(i, inplace = True)
        else:
            words = data.content[i][0]
            for k in range(len(data.content[i])-1):
                words+= " " + data.content[i][k+1]
            data.content[i] = words
            data.content[i] = re.sub(r'[^\w]', ' ', data.content[i])
            if len(data.content[i]) == 0:
                data.drop(i, inplace = True)
            else:
                data.content[i] = ''.join(''.join(s)[:2] for _, s in itertools.groupby(data.content[i]))
                data.content[i] = data.content[i].replace("'", "")
                data.content[i] = nltk.tokenize.word_tokenize(data.content[i])
                #data.content[i] = [w for w in data.content[i] if not w in stopset]
                for j in range(len(data.content[i])):
                    data.content[i][j] = lem.lemmatize(data.content[i][j], "v")
                if len(data.content[i]) == 0:
                    data.drop(i, inplace = True)

        
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

vectorizer = TfidfVectorizer(min_df=3, max_df=0.9)

train_vectors = vectorizer.fit_transform(x_train)
test_vectors = vectorizer.transform(x_test)

model = svm.SVC(kernel='linear') 
model.fit(train_vectors, y_train) 
predicted_sentiment = model.predict(test_vectors)

print(classification_report(y_test, predicted_sentiment))

predicted_sentiments = []
for s in range(len(predicted_sentiment)):
    predicted_sentiments.append(predicted_sentiment[s])
    
prediction_df = pd.DataFrame({'Content':x_test, 'Emotion_predicted':predicted_sentiment, 'Emotion_actual': y_train})
prediction_df.to_csv('emotion_recognizer.csv', index = False)

elapsed_time = time.time() - start_time
print ("processing time:", elapsed_time, "seconds")
