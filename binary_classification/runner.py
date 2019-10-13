'''
Runner module
Runs against user input to classify text as belonging to 'disease'

@author Disha Misal
'''

import json
from keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer

# User Configuration
TOKENS_FILE = "tokens.json"
MODEL_FILE = 'binary_sequential_trained_model.h5'

def classify(predict_val, separation_pt):
	print(str(predict_val < separation_pt) + ' : ' + str(predict_val))

with open(TOKENS_FILE) as file:
	sentences = json.loads(file.read())

vectorizer = CountVectorizer()
vectorizer.fit(sentences)

model = load_model(MODEL_FILE)
separation_pt = model.predict(vectorizer.transform(["nonexistentstringforgoodmeasure"]), verbose=False)

while True:
	user_input = vectorizer.transform([raw_input('Text to categorize as disease: ')])
	predict_val = model.predict(user_input, verbose=False)[0][0]
	classify(predict_val, separation_pt)
