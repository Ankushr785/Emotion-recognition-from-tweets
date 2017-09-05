#important libraries
import pandas as pd
import numpy as np
import nltk
import re
#importing stopwords is optional, in this case it decreased accuracy
#from nltk.corpus import stopwords
import itertools
import time
from sklearn.metrics import classification_report


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
    
x = int(np.round(len(data)*0.75))
train = data.iloc[:x,:].reset_index(drop = True)
test = data.iloc[x:,:].reset_index(drop = True)
    
from textblob.classifiers import NaiveBayesClassifier as NBC

training_corpus = []

for k in range(len(train)):
    training_corpus.append((train.content[k], train.sentiment[k]))
    
test_corpus = []

for l in range(len(test)):
    test_corpus.append((test.content[l], test.sentiment[l]))

model = NBC(training_corpus)

print(model.accuracy(test_corpus))

predictions = []
for m in range(len(test)):
    predictions.append(model.classify(test.content[m]))
print(classification_report(test.sentiment, predictions)
    
predictions_df = pd.DataFrame({'Content':test.content, 'Emotion_predicted':predictions, 'Emotion_actual':test.sentiment})
predictions_df.to_csv('naive_emotion_recognizer.csv', index = False)

elapsed_time = time.time() - start_time
print ("processing time:", elapsed_time, "seconds")
