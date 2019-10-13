'''
HTML parser module
Reads html documents and extracts sentences to store in a json

@author Disha Misal
'''

import json
import pandas as pd
from os import listdir
from bs4 import BeautifulSoup
from itertools import chain

# User Configuration
POS_PATH = "training_set/positive"
TRUNC_LIMIT = 25 # files per set
DISEASES_FILE = "diseases.json"

def write_to_json(disease_map):
	with open(DISEASES_FILE, 'w') as file:
		file.write(json.dumps({"diseases" : disease_map}))

# Driver
if (__name__ == '__main__'):

	disease_map = {}
	files = [f for f in listdir(POS_PATH)][:TRUNC_LIMIT]

	for f in files:
		file_name = POS_PATH + '/' + f
		sentence_list = []
		with open(file_name) as html_file:
			html = html_file.read()
			soup = BeautifulSoup(html)
			sentence_list += list(soup.findAll("body")[0].contents[5].stripped_strings)
		disease_map[f] = sentence_list

	write_to_json(disease_map)