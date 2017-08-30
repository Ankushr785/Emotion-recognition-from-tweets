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

> A look at the data before and after pre-processing
