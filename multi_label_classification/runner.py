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
TOKENS_FILE = "tokens.json"
MODEL_FILE = "multi_class_sequential_trained_model.h5"
DISEASES_FILE = "diseases.json"

def output_score(predicted_label):
	print("Predicted Label: ", predicted_label)

with open(TOKENS_FILE) as file:
	sentences = json.loads(file.read())

vectorizer = CountVectorizer()
vectorizer.fit(sentences)

model = load_model(MODEL_FILE)

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

while True:
	predicted_label = le.inverse_transform([np.argmax(model.predict(vectorizer.transform([raw_input('Text to categorize as disease: ')]), verbose=False))])[0]
	output_score(predicted_label)
