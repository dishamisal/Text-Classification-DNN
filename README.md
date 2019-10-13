# Classify Disease Articles using Deep Neural Networks

[![N|Solid](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRXSSgSZw-jLD3ScEZLqxo93UDBkZJ-CDtUL7l_uIJaxTVvSX3d)](https://www.tensorflow.org/guide/keras)

## Problem Statement
Classify whether an article/text describes an disease. Neural-network variants:
1. Binary Classification of whether an article accurately describes an disease
2. Figure out the disease it might be referring to

## Source Dataset
Obtained from Wikipedia by scraping through articles
- Gather articles pertaining to diseases and otherwise using wget
- Label each article depending on if it pertains to an disease: isDisease
- HTML Parser to scrape through the essentials from the html document

## Logistic Regression (baseline)
LogisticRegression from Scikit-learn is used for:
- Feature extraction and transformation
- Logistic Regression Classifier is used to train on the dataset and test on the testing dataset

## Binary Classification using Deep Neural Networks
Keras with tensorflow as the backend and scikit-learn for feature extraction:
- Sentences are extracted from the article and vectorized using [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) from the scikit-learn library
- [Sequential](https://keras.io/models/sequential/) deep neural network model with 10 layers with relu activation and adam optimizer is used to train on the data
- Verification is accomplished by splitting the dataset into training and test datasets

## Multi-label Disease classification using Deep Neural Networks
Keras with tensorflow as the backend and scikit-learn for feature extraction:
- Vectorized sentences are tagged along with the disease labels
- Labels are [LabelEncoded](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) and transformed into a [OneHotVector](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) to be processed by the DNN model
- Sequential deep neural network model with 10 layers and output layer with multiple classes is used to train on the data

## Runner
Models created for both the parts are trained for sample data-sets and stored as .h5 files
> Using these, runner modules could be leveraged to provide user with a script to test out DNN model on-demand

```sh
$ cd {Part}/trained_models
$ python runner.py
$ (enter text to be classified)
```

## Conclusions
Model performs satisfactorily well, but with caveats. Future scope includes:
- Experimentation with Word embeddings and Glove bag-of-words
- Convolutional Neural Networks and Deep-NLP

