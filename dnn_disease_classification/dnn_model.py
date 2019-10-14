'''
Binary Sequential DNN
Trains and save the DNN model
Reference: https://towardsdatascience.com/text-classification-in-python-dd95d264c802

@author Disha Misal
'''

import json
import time
import pandas as pd
from os import listdir
from bs4 import BeautifulSoup
from itertools import chain

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras import layers

# User Configuration
TOKENS_FILE = "tokens.json"
BIN_MODEL_FILE = "binary_sequential_trained_model.h5"
MULTI_MODEL_FILE = "multi_label_sequential_trained_model.h5"
DISEASES_FILE = "diseases.json"
POS_TOKENS = "positive_sentences.json"
NEG_TOKENS = "negative_sentences.json"
VERBOSE = True

def merge_into_map(str_list, word_map):
	for s in str_list:
		if s in word_map:
			word_map[s] += 1
		else:
			word_map[s] = 1

def save_models(bin_model, multi_model):
	bin_model.save(BIN_MODEL_FILE)
	multi_model.save(MULTI_MODEL_FILE)

def save_tokens_and_vocabulary(vectorizer):
	with open(TOKENS_FILE, 'w') as file:
		file.write(json.dumps(vectorizer.vocabulary_))

# Driver
if __name__=="__main__":
	start = time.time()
	
	'''
	Binary classification for classifying if the text/article relates to a disease
	Sequential deep neural-network model
	'''

	positive_sentences = []
	neative_sentences = []
	with open(POS_TOKENS) as file:
		positive_sentences = json.loads(file.read())['positive']
	with open(NEG_TOKENS) as file:
		negative_sentences = json.loads(file.read())['negative']

	df = pd.concat([
		pd.DataFrame(list([(w,0) for w in positive_sentences]), columns=['Sentence', 'Label']),
		pd.DataFrame(list([(w,1) for w in negative_sentences]), columns=['Sentence', 'Label'])
	])

	sentences = df['Sentence'].values
	labels = df['Label'].values
	sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.01, random_state=1000)

	vectorizer = CountVectorizer()
	vectorizer.fit(sentences)

	x_train = vectorizer.transform(sentences_train)
	x_test  = vectorizer.transform(sentences_test)
	input_dim = x_train.shape[1]

	'''
	Baseline:
	Logistic Regression Classifier
	'''

	classifier = LogisticRegression()
	classifier.fit(x_train, y_train)
	score = classifier.score(x_test, y_test)
	print("Baseline LogisticRegression Accuracy: {:.4f}".format(score))

	bin_model = Sequential()
	bin_model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
	bin_model.add(layers.Dense(1, activation='sigmoid'))

	# To print bin_model: bin_model.summary()
	bin_model.compile(loss='binary_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])

	bin_model.fit(x_train, y_train,
                    epochs=100,
                    verbose=False,
                    validation_data=(X_test, y_test),
					batch_size=10)

	loss, accuracy = bin_model.evaluate(x_train, y_train, verbose=False)
	print("Binary Model Training Accuracy: {:.4f}".format(accuracy))
	loss, accuracy = bin_model.evaluate(x_test, y_test, verbose=False)
	print("Binary Model Testing Accuracy:  {:.4f}".format(accuracy))
	
	'''
	Multi-label classification for identifying the disease
	Sequential deep neural-network model
	'''

	disease_map = {}
	with open(DISEASES_FILE) as file:
		disease_map = json.loads(file.read())['diseases']
	distinct_labels = len(disease_map.keys())

	df = []
	for disease, sentences in disease_map.iteritems():
		df.append(pd.DataFrame(list([(w,disease) for w in sentences]), columns=['Sentence', 'Label']))
	df = pd.concat(df)

	# Transform to OneHotArray
	sentences = df['Sentence'].values
	labels = df['Label'].values
	le = LabelEncoder()
	label_encoded = le.fit_transform(labels)
	label_encoded = label_encoded.reshape(len(label_encoded), 1)
	y = OneHotEncoder().fit_transform(label_encoded)

	sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.01, random_state=1000)

	vectorizer = CountVectorizer()
	vectorizer.fit(sentences)

	x_train = vectorizer.transform(sentences_train)
	x_test  = vectorizer.transform(sentences_test)
	input_dim = x_train.shape[1]

	multi_model = Sequential()
	multi_model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
	multi_model.add(layers.Dense(distinct_labels, activation='sigmoid'))

	multi_model.compile(loss='binary_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])

	multi_model.fit(x_train, y_train,
                    epochs=100,
                    verbose=False,
                    validation_data=(x_test, y_test),
					batch_size=10)
	
	loss, accuracy = multi_model.evaluate(x_train, y_train, verbose=False)
	print("Multi-label Model Training Accuracy: {:.4f}".format(accuracy))
	loss, accuracy = multi_model.evaluate(x_test, y_test, verbose=False)
	print("Multi-label Model Testing Accuracy:  {:.4f}".format(accuracy))
	
	save_models(bin_model, multi_model)
	save_tokens_and_vocabulary(vectorizer)

	if VERBOSE:
		end = time.time()
		elapsed = float(end - start)/float(60*60)
		print('Time: ', elapsed)
