'''
Multi-label Sequential DNN
Trains and save the DNN model

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
MODEL_FILE = "multi_label_sequential_trained_model.h5"
DISEASES_FILE = "diseases.json"
VERBOSE = True

def merge_into_map(str_list, word_map):
	for s in str_list:
		if s in word_map:
			word_map[s] += 1
		else:
			word_map[s] = 1

def save_model(model):
	model.save(MODEL_FILE)

def save_tokens_and_vocabulary(vectorizer):
	with open(TOKENS_FILE, 'w') as file:
		file.write(json.dumps(vectorizer.vocabulary_))

# Driver
if __name__=="__main__":
	start = time.time()

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

	X_train = vectorizer.transform(sentences_train)
	X_test  = vectorizer.transform(sentences_test)
	input_dim = X_train.shape[1]

	model = Sequential()
	model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
	model.add(layers.Dense(distinct_labels, activation='sigmoid'))

	model.compile(loss='binary_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
	history = model.fit(X_train, y_train,
                    epochs=100,
                    verbose=False,
                    validation_data=(X_test, y_test),
					batch_size=10)
	
	loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
	print("Training Accuracy: {:.4f}".format(accuracy))
	loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
	print("Testing Accuracy:  {:.4f}".format(accuracy))

	save_model(model)
	save_tokens_and_vocabulary(vectorizer)

	if VERBOSE:
		end = time.time()
		elapsed = float(end - start)/float(60*60)
		print('Time: ', elapsed)
