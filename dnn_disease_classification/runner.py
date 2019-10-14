'''
Runner module
Runs against user input to classify text as belonging to 'disease'

@author Disha Misal
'''

import json
from keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# User Configuration
TOKENS_FILE = "pretrained_model/tokens.json"
DISEASES_FILE = "pretrained_model/diseases.json"
BIN_MODEL_FILE = "pretrained_model/binary_sequential_trained_model.h5"
MULTI_MODEL_FILE = "pretrained_model/multi_label_sequential_trained_model.h5"

def classify(predict_val, separation_pt):
	return predict_val <= separation_pt

def output_score(predicted_label):
	print("Label: " + predicted_label.encode("utf-8"))

with open(TOKENS_FILE) as file:
	sentences = json.loads(file.read())

vectorizer = CountVectorizer()
vectorizer.fit(sentences)

df = []
disease_map = {}
with open(DISEASES_FILE) as file:
	disease_map = json.loads(file.read())['diseases']
	for disease, sentences in disease_map.iteritems():
		df.append(pd.DataFrame(list([(w,disease) for w in sentences]), columns=['Sentence', 'Label']))
df = pd.concat(df)

labels = df['Label'].values
le = LabelEncoder()
label_encoded = le.fit_transform(labels)

bin_model = load_model(BIN_MODEL_FILE)
multi_model = load_model(MULTI_MODEL_FILE)
separation_pt = bin_model.predict(vectorizer.transform(["nonexistentstringforgoodmeasure"]), verbose=False)

while True:
	user_str = raw_input('>>> Text to categorize as disease: ')
	user_input = vectorizer.transform([user_str])
	predict_val = bin_model.predict(user_input, verbose=False)[0][0]
	if classify(predict_val, separation_pt):
		predicted_label = le.inverse_transform([np.argmax(multi_model.predict(vectorizer.transform([user_str]), verbose=False))])[0]
		output_score(predicted_label)
	else:
		print("Not a disease. Try again!")
