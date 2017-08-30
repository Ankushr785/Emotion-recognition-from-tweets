# Emotion-recognition-from-tweets
A comprehensive approach on recognizing emotion (sentiment) from a certain tweet. Supervised machine learning.

**Problem Statement**

-> Given a dataset mapping tweets to the associated emotions, an emotion recognizer algorithm needs to be created.

-> Data source - 

-> Libraries - Natural Language Tool-kit (NLTK) and Sci-kit learn 

**Pre - processing**

-> Removal of regular expressions, symbols using the 're' library

-> Removal of lemmas (Lexicon Normalization) using WordNetLemmatizer from NLTK

-> Removal of multi-letter ambiguities, e.g 'noooo' gets converted to 'no'

-> (Optional) Removal of stop-words  - caused decrease in f1-score as well as overall accuracy

> A look at the data before and after pre-processing

*before*

![before_pre](https://user-images.githubusercontent.com/26039458/29850620-868992a4-8d4c-11e7-95f2-582be11a7bbd.png)

*after*

![after](https://user-images.githubusercontent.com/26039458/29850682-f58c4f20-8d4c-11e7-821a-b2cf3ccb4e88.png)

**Vectorization**

-> Term frequency - inverse document frequency (TfidfVectorizer) deployed for converting the words to vectors

**Model - 1**

-> Support Vector Machine

-> Performance
