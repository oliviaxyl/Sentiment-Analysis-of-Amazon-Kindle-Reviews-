# Sentiment-Analysis-of-Amazon-Kindle-Reviews-

![sentimentanalysis](https://user-images.githubusercontent.com/49653689/94883213-10b66280-0438-11eb-9eda-f0288f6f91ed.png)

#### -- Project Status: [Completed]

## Project Objective
The purpose of this project is to help customer service team to get the “feeling of a message" without reading every single word. This saves the agent from needing to read the entire reviews and let them focus their time and effort on solving issues within their capabilities. 

### Methods Used
* Text preprocessing 
* Word Vectorization (BoW, TF-IDF, and One-Hot Encoding) => Classifiers (SVM, MultinomialNB and Artificial Neural Network)
* Transfer Learning: LSTM & GRU with GloVe embedding
* Fine-tuned BERT language model

### Technologies

* Jupyter, Python 3
* Pandas, Numpy, NLTK, Scikit-learn, TensorFlow, keras, PyTorch, seaborn, matplotlib

## Project Description

### Data 

A small subset of product reviews in [Amazon reviews: Kindle Store Category](https://www.kaggle.com/bharadwaj6/kindle-reviews/notebooks) obtained from [Amazon Review Data (2018)](https://nijianmo.github.io/amazon/index.html).

### Text Processing

* Convert rating 1 & 2 to 'Negative', 3 to 'Neutral', and 4 & 5 to 'Positive'.
* Under-sample 'Positive', 'Neutral', and 'Negative' sentimens ratio to 1:1:1.
* Replace contraction (eg. "I'm" -> "I am", "let's" -> "let us", "shouldn't've" -> "should not have")
* Removed stopwords (eg. 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how')
* Find POS tag -> Lemmatize tokens (eg. ‘Caring’ -> ‘Care’, 'better' -> 'good',  'asked' -> 'ask' )
* Remove URL ('http\S+')
* Remove emoji (eg. :nerd_face: )

### Word Embedding

* Bag of words - 'CountVectorizer'
* One-Hot - 'Binarizer'
* TF-IDF - 'TfidfVectorizer'

### Text Classifiers

* Machine Learning: SVM, MultinomialNB

* Neural Network: ANN

* RNN: LSTM, GRU

* Language Model: BERT

### Test Accuracy

* One-Hot & SVM (Baseline) -- > 61.82%
* One-Hot & MultinomialNB -- > 72.53%
* Bag of Words & MultinomialNB -- > 71.36%
* TF-IDF & MultinomialNB -- > 72.88%
* TF-IDF & Artificial Neural Network -- > 71.01%
* GloVe embedding & LSTM -- > %
* GloVe embedding & GRU -- > %
* Fine-tuning BERT -- > 76.72%

## Reference

Understanding NLP Word Embeddings — Text Vectorization 
https://towardsdatascience.com/understanding-nlp-word-embeddings-text-vectorization-1a23744f7223

Natural Language Processing: From Basics to using RNN and LSTM 
https://towardsdatascience.com/natural-language-processing-from-basics-to-using-rnn-and-lstm-ef6779e4ae66

Word Embedding Explained, a comparison and code tutorial 
https://medium.com/@dcameronsteinke/tf-idf-vs-word-embedding-a-comparison-and-code-tutorial-5ba341379ab0

Understanding Word Embeddings with TF-IDF and GloVe 
https://towardsdatascience.com/understanding-word-embeddings-with-tf-idf-and-glove-8acb63892032

Illustrated Guide to LSTM’s and GRU’s: A step by step explanation 
https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21

Simple LSTM for text classification 
https://www.kaggle.com/kredy10/simple-lstm-for-text-classification

NLP using GloVe Embeddings 
https://www.kaggle.com/madz2000/nlp-using-glove-embeddings-99-87-accuracy

Transfer Learning for NLP: Fine-Tuning BERT for Text Classification
https://www.analyticsvidhya.com/blog/2020/07/transfer-learning-for-nlp-fine-tuning-bert-for-text-classification/

Multi Class Text Classification With Deep Learning Using BERT
https://towardsdatascience.com/multi-class-text-classification-with-deep-learning-using-bert-b59ca2f5c613

A Visual Guide to Using BERT for the First Time
http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/

Tokenizers: How machines read 
https://blog.floydhub.com/tokenization-nlp/

3 subword algorithms help to improve your NLP model performance
https://medium.com/@makcedward/how-subword-helps-on-your-nlp-model-83dd1b836f46

Text preprocessing for BERT
https://www.kaggle.com/c/google-quest-challenge/discussion/127881

The Illustrated Transformer
http://jalammar.github.io/illustrated-transformer/

BERT Fine-Tuning Sentence Classification
https://colab.research.google.com/drive/1Y4o3jh3ZH70tl6mCd76vz_IxX23biCPP#scrollTo=nSU7yERLP_66

BERT Explained – A list of Frequently Asked Questions
https://yashuseth.blog/2019/06/12/bert-explained-faqs-understand-bert-working/

 
