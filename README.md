# Emotion-recognition-from-tweets
A comprehensive approach on recognizing emotion (sentiment) from a certain tweet. Supervised machine learning.
The 'Untitled.ipynb' file consists of a jupyter notebook of test codes.
Rest of the .py files consist of code corresponding to their names.

Find the complete explanation to the approach here https://medium.com/@ankushraut/artificial-neural-network-for-text-classification-b7aa5994d985.

**Problem Statement**

-> Given a dataset mapping tweets to the associated emotions, an emotion recognizer algorithm needs to be created.

-> Data source - https://l.facebook.com/l.php?u=https%3A%2F%2Fwww.crowdflower.com%2Fwp-content%2Fuploads%2F2016%2F07%2Ftext_emotion.csv&h=ATMfMjsluVtxjiGHelTFi-X_nKckyiEYJHwiGN9u-1LriOzSoQ4oFbjCBZItdFCummCAEnLlD_m7bUwMZs4LrZnDgDt4txt3laalwYWZnETPZBqChySF_gURrBRyFNGFNb_y&s=1&enc=AZP8Ci-JshpS1aE5ggdBCNoVTeDCTGhQTic6bXTQJ_M6PcwGzRMJfqiPYywc62pbkKEwvWl-M_9_OcEzpf1lKZ38b8xGHiCDtUf50JN-W6vA_Q

-> Libraries - Natural Language Tool-kit (NLTK) and Sci-kit learn 

**Pre - processing**

-> Removal of regular expressions, symbols using the 're' library

-> Removal of lemmas (Lexicon Normalization) using WordNetLemmatizer from NLTK

-> Removal of multi-letter ambiguities, e.g 'noooo' gets converted to 'no'

-> (Optional) Removal of stop-words  - caused decrease in f1-score as well as overall accuracy

> A look at the data before and after pre-processing

*before*

![whole_before](https://user-images.githubusercontent.com/26039458/29850795-c896e38a-8d4d-11e7-932a-e3e30a9ad675.png)

*after*

![whole_after](https://user-images.githubusercontent.com/26039458/29850797-cce512fe-8d4d-11e7-84c5-229422f0e142.png)


**Vectorization**

-> Term frequency - inverse document frequency (TfidfVectorizer) deployed for converting the words to vectors (for SVM and Naive Bayes)

-> Bag of words representation used as an input for the sigmoid layers model

**Model - 1**

-> Support Vector Machine - Creation of hyperplanes separating all the classes, linear kernel.

**Model - 2**

-> Naive Bayes classifier - naively assuming no inter-dependence between words of a sentence corpus.

**Model -3**

-> Aritificial Neural Network - 3 layer neural network with sigmoid activation and gradient descent optimization
