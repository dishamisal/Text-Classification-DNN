'''
HTML parser
Reads html documents and extracts sentences to store in a json with binary tags

@author Disha Misal
'''

import json
import pandas as pd
from os import listdir
from bs4 import BeautifulSoup

# User Configuration
POS_PATH = "training/positive"
NEG_PATH = "training/positive"
TRUNC_LIMIT = 50 # files per set
DISEASES_FILE = "diseases.json"
POS_TOKENS = "positive_sentences.json"
NEG_TOKENS = "negative_sentences.json"


def write_to_json(positive_sentences, negative_sentences, disease_map):
	with open(POS_TOKENS, 'w') as file:
		file.write(json.dumps({"positive" : positive_sentences}))
	with open(NEG_TOKENS, 'w') as file:
		file.write(json.dumps({"negative" : negative_sentences}))
	with open(DISEASES_FILE, 'w') as file:
		file.write(json.dumps({"diseases" : disease_map}))
	return

# Driver
if (__name__ == '__main__'):
	
	positive_sentences, negative_sentences = [], []
	disease_map, negatives_map = {}, {}
	positive_files = [f for f in listdir(POS_PATH)][:TRUNC_LIMIT]
	negative_files = [f for f in listdir(NEG_PATH)][:TRUNC_LIMIT]

	consume_sections = [
		(positive_files, POS_PATH, positive_sentences, disease_map),
		(negative_files, NEG_PATH, negative_sentences, negatives_map)
	]

	for i, (files, path, sentence_list, label_map) in enumerate(consume_sections):
		for f in files:
			file_name = path + '/' + f
			with open(file_name) as html_file:
				html = html_file.read()
				soup = BeautifulSoup(html) # Todo: Clean strings
				## Extract the main body from the html page ##
				sentence_list += list(soup.findAll("body")[0].contents[5].stripped_strings)
			label_map[f.decode("latin-1").encode("utf-8")] = sentence_list

	write_to_json(positive_sentences, negative_sentences, disease_map)
	