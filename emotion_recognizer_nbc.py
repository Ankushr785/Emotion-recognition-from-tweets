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
os.chdir('/tmp/guest-pltjjp/Downloads')


data = pd.read_csv('text_emotion.csv')
data = data.iloc[:1000,:]


#stopset = set(stopwords.words('english'))

from nltk.stem.wordnet import WordNetLemmatizer 
lem = WordNetLemmatizer()

#comprehensive cleaning
def cleaning(text):
    txt = str(text)
    txt = re.sub(r"http\S+", "", txt)
    if len(txt) == 0:
        return 'no text'
    else:
        txt = txt.split()
        index = 0
        for j in range(len(txt)):
            if txt[j][0] == '@':
                index = j
        txt = np.delete(txt, index)
        if len(txt) == 0:
            return 'no text'
        else:
            words = txt[0]
            for k in range(len(txt)-1):
                words+= " " + txt[k+1]
            txt = words
            txt = re.sub(r'[^\w]', ' ', txt)
            if len(txt) == 0:
                return 'no text'
            else:
                txt = ''.join(''.join(s)[:2] for _, s in itertools.groupby(txt))
                txt = txt.replace("'", "")
                txt = nltk.tokenize.word_tokenize(txt)
                #data.content[i] = [w for w in data.content[i] if not w in stopset]
                for j in range(len(txt)):
                    txt[j] = lem.lemmatize(txt[j], "v")
                if len(txt) == 0:
                    return 'no text'
                else:
                    return txt
                
data['content'] = data['content'].map(lambda x: cleaning(x))


        
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

from sklearn.metrics import classification_report

predictions = []
for m in range(len(test)):
    predictions.append(model.classify(test.content[m]))
print(classification_report(test.sentiment, predictions))
    
predictions_df = pd.DataFrame({'Content':test.content, 'Emotion_predicted':predictions, 'Emotion_actual':test.sentiment})
predictions_df.to_csv('naive_emotion_recognizer.csv', index = False)

elapsed_time = time.time() - start_time
print ("processing time:", elapsed_time, "seconds")
