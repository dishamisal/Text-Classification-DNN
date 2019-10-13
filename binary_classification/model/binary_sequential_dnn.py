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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from keras.models import Sequential
from keras import layers

# User Configuration
TOKENS_FILE = "tokens.json"
MODEL_FILE = "binary_sequential_trained_model.h5"
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
	
	positive_sentences = []
	neative_sentences = []
	with open('positive_sentences.json') as file:
		positive_sentences = json.loads(file.read())['positive']
	with open('negative_sentences.json') as file:
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

	model = Sequential()
	model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
	model.add(layers.Dense(1, activation='sigmoid'))

	# To print model: model.summary()
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
