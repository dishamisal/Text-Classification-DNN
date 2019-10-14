'''
Logistic Regression Classifier

@author Disha Misal
'''

import json
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def print_score(score):
	print('LogisticRegression accuracy: ',score)

# Driver
if __name__=="__main__":
	
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
	sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.25, random_state=1000)

	vectorizer = CountVectorizer()
	vectorizer.fit(sentences)

	x_train = vectorizer.transform(sentences_train)
	x_test  = vectorizer.transform(sentences_test)

	classifier = LogisticRegression()
	classifier.fit(x_train, y_train)
	score = classifier.score(x_test, y_test)
	
	print_score(score)
