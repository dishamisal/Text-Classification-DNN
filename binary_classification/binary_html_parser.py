'''
Binary HTML parser
Reads html documents and extracts sentences to store in a json with binary tags

@author Disha Misal
'''

import json
import pandas as pd
from os import listdir
from bs4 import BeautifulSoup

# User Configuration
POS_PATH = "training_set/positive"
NEG_PATH = "training_set/negative"
TRUNC_LIMIT = 50 # files per set


def write_to_json(positive_sentences, negative_sentences):
	with open('positive_sentences.json', 'w') as file:
		file.write(json.dumps({"positive" : positive_sentences}))
	with open('negative_sentences.json', 'w') as file:
		file.write(json.dumps({"negative" : negative_sentences}))
	return

# Driver
if (__name__ == '__main__'):
	
	positive_sentences, negative_sentences = [], []
	positive_files = [f for f in listdir(POS_PATH)][:TRUNC_LIMIT]
	negative_files = [f for f in listdir(NEG_PATH)][:TRUNC_LIMIT]

	consume_sections = [
		(positive_files, POS_PATH, positive_sentences),
		(negative_files, NEG_PATH, negative_sentences)
	]

	for i, (files, path, sentence_list) in enumerate(consume_sections):
		for f in files:
			file_name = path + '/' + f
			with open(file_name) as html_file:
				html = html_file.read()
				soup = BeautifulSoup(html) # Todo: Clean strings
				sentence_list += list(soup.findAll("body")[0].contents[5].stripped_strings)

	write_to_json(positive_sentences, negative_sentences)
	