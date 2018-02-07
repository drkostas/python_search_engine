#!/usr/bin/python
# BuildIndex: Assembly the Inverted Index. Bear in mind that inverted index is the data structure that maps tokens to the documents they appear in.
# http://aakashjapi.com/fuckin-search-engines-how-do-they-work/

import os
import re
import sys
import json
import time
import argparse
import math
import operator

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.wordnet import WordNetLemmatizer



wp_tokenizer = WordPunctTokenizer()			# Tokenizer instance
wnl_lemmatizer = WordNetLemmatizer()		# Wordnet Lemmatizer instance
stop_words = stopwords.words('english')		# English stop words list
inverted_file = {}							# The Inverted File data structure
docnames = []								# The document names in the order they've been found

total_doc_cnt = 0							# Total number of indexed documents
indexed_words = 0							# Total (corpus) number of indexed terms
excluded_words = 0							# Total (corpus) number of exluded terms



def set_argParser():
	""" The build_index script's arguments presentation method."""
	argParser = argparse.ArgumentParser(description="Script's objective is to assembly the inverted index of a given document collection.")
	argParser.add_argument('-I', '--input_dir', type=str, default=os.path.dirname(os.path.realpath(__file__)) + os.sep + 'books', help='The directory path of the document collection. Default:' + os.path.dirname(os.path.realpath(__file__)) + os.sep + 'books')
	argParser.add_argument('-O', '--output_dir', default=os.path.dirname(os.path.realpath(__file__)), type=str, help='The output directory path where the inverted file is going to be exported in JSON format. Default: (' + os.path.dirname(os.path.realpath(__file__)))

	return argParser



def check_arguments(argParser):
	""" Parse and check the inserted command line args."""
	line_args = argParser.parse_args()

	# 'input_dir' line argument handling
	if (not(os.path.exists(os.path.realpath(line_args.input_dir)))) :
		line_args.input_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep + 'books'
	if (not(line_args.input_dir.endswith(os.sep))):
		line_args.input_dir += os.sep

	# 'output_dir' line argument handling
	if (not(os.path.exists(os.path.realpath(line_args.output_dir)))) :
		line_args.output_dir = os.path.dirname(os.path.realpath(__file__))
	if (not(line_args.output_dir.endswith(os.sep))):
		line_args.output_dir += os.sep

	return line_args



def export_output(line_args):
	""" Export the Inverted File structure to a JSON file."""
	# http://stackoverflow.com/questions/12309269/how-do-i-write-json-data-to-a-file-in-python
	json_file = line_args.output_dir + 'inverted_file.txt'
	with open(json_file, 'w') as fh:
		json.dump(inverted_file, fh,separators = (',',':'))

	json_file = line_args.output_dir + 'docnames.txt'
	with open(json_file, 'w') as fh:
		json.dump(docnames, fh,separators = (',',':'))

def build_inverted():
	""" Recreating Inverted File with 3-in-4 Front Coding. """
	global inverted_file

	inverted_file2 = []							# Temporary List to chang the form of the Inverted File 
	for word,dic in inverted_file.iteritems():
		inverted_file2.append([word,dic])
	inverted_file = inverted_file2

	for i in range(len(inverted_file)):			# Adding two additional values to each word, the number of letters sliced and the distance from the previous whole word.
		inverted_file[i] = inverted_file[i][:1] + [0,0] + inverted_file[i][1:]

	inverted_file = sorted(inverted_file, key = operator.itemgetter(0)) # Sorting alphabetically

	# ------------------- Slicing the letters ------------------- #
	for i in range(0,len(inverted_file),4):
		startword = inverted_file[i]
		for j in range(1,4):
			if i+j+1 > len(inverted_file):
				break

			letter = 0
			wordtocheck = inverted_file[i+j]
			inverted_file[i+j][2] = j
			while(1): # Slicing letters from the beggining of the word(j) until we run out of letters(i or j) or we won't find the same letter on both words.
				if (len(inverted_file[i+j][0])>0) and (letter<len(startword[0])):
					if (startword[0][letter] == inverted_file[i+j][0][0]):
						letter += 1
						inverted_file[i+j][1] += 1
						inverted_file[i+j][0] = inverted_file[i+j][0][1:]
					else:
						break
				else:
					break
	# Inverted File Form example for the words find,finished, finland and go:
	# [ 
	# 	[ "find",0,0,{"document  number":[ [ positions in doc ], tf*idf ]} ],
	# 	[ "ished",3,1,{"document  number":[ [ positions in doc ], tf*idf ]} ],
	# 	[ "land",3,2,{"document  number":[ [ positions in doc ], tf*idf ]}],
	# 	[ "go",0,3,{"document  number":[ [ positions in doc ], tf*idf ]}] 
	# ]

def calculate_tfidf():
	""" Calculate the TF * IDF per lemma."""
	global inverted_file

	for lemma in inverted_file.keys():
		# Inverted document frequency = 10000*log(Total number of documents / Number of documents appeared)   ---I multiply idf by 10000 for normalization---
		idf = math.log(float(total_doc_cnt) / len(inverted_file[lemma].keys())) if len(inverted_file[lemma].keys())>0 else 0
		idf = round(idf*100000)

		for docid in inverted_file[lemma].keys():
			# Inverted List structure:
			#    <key>    :                              <value>
			# Document id : (Term's frequency, [Term's order of appearance list], Tf * IDf)
			inverted_file[lemma][docid].append(int(inverted_file[lemma][docid][0] * idf))
			del inverted_file[lemma][docid][0]

def update_inverted_index(existing_lemmas,word_cnt):
	""" Update the Inverted File structure.."""
	global inverted_file
	for lemma in existing_lemmas.keys():
		if(lemma not in inverted_file.keys()):
			inverted_file[lemma] = {}
		tf = float(len(existing_lemmas[lemma]))/word_cnt
		inverted_file[lemma][docid] = [tf, existing_lemmas[lemma]]
	# Current Inverted File Structure [for each word]:
	# { Document 1: [[Where it appears],tf*idf], ... , Document n: [[Where it appears],tf*idf] }

if (__name__ == "__main__") :
	argParser = set_argParser()				# The argument parser instance
	line_args = check_arguments(argParser)	# Check and redefine, if necessary, the given line arguments 

	# ------------------------------------------------------------------------------- #
	# 								Text File Parsing								  #
	# ------------------------------------------------------------------------------- #
	for file in os.listdir(line_args.input_dir):
		if (not(file.endswith(".txt"))):					# Skip anything but .txt files
			continue
		total_doc_cnt += 1									# Increment the total number of processed documents
		docnames.append(re.sub(r'\.txt$', '', file))		# Document's ID -String-
		docid = total_doc_cnt								# Document's ID -Integer-
		existing_lemmas = {}								# Dictionary with the document's lemmas

		with open(line_args.input_dir + file, "r") as fh:
			tick = time.time()
			print "Processing: " + line_args.input_dir + file,

			word_cnt = 0 		# Our inverted index would map words to document names but, we also want to support phrase queries: queries for not only words, 
								# but words in a specific sequence => We need to know the order of appearance.

			for line in fh:
				for word, pos in pos_tag(wp_tokenizer.tokenize(line.lower().strip())):
					if(
						re.search(r'[\W_]+', word) or 	# If includes a non-letter character
						word in stop_words or			# If this is a stop word
						# http://stackoverflow.com/questions/15388831/what-are-all-possible-pos-tags-of-nltk
						#   CC: conjuction, coordinating
						#   LS: List item marker
						#   EX: Existential there
						#   MD: Modal auxiliary
						#  PDT: Pre-determined
						#  PRP: Pronoun, personal
						# PRP$: Pronoun, possesive
						#  WDT: WH-determiner
						#   WP: WH-pronoun
						#  WRB: Wh-adverb
						pos in ['CC', 'LS', 'EX', 'MD', 'PDT', 'PRP', 'PRP$', 'WDT', 'WP', 'WRB']
					):
						excluded_words += 1
						continue

					pos = 'v' if (pos.startswith('VB')) else 'n'									# If current term's appearance is verb related then 
																									# the POS lemmatizer should be verb ('v'), otherwise ('n')

					lemma = wnl_lemmatizer.lemmatize(word, pos)										# Stemming/Lemmatization

					if (lemma not in existing_lemmas):
						existing_lemmas[lemma] = []

					if not existing_lemmas[lemma]:
						existing_lemmas[lemma].append(word_cnt)									# Keep lemma's current position
					else:
						existing_lemmas[lemma].append(word_cnt - sum(existing_lemmas[lemma]))
					word_cnt += 1																	# Increment the position pointer by 1
					indexed_words += 1																# Increment the total indeces words count
			# Update the Inverted File structure with current document information
			update_inverted_index(existing_lemmas,word_cnt)

			print "({0:>6.2f} sec)".format(time.time() - tick)
	# ------------------------------------------------------------------------------------------- #

	calculate_tfidf()			# Enrich the Inverted File structure with the Tf*IDf information
	tick = time.time()
	print "Building Inverted File...",
	build_inverted()			# Making Inverted File a List, Sorting and Front Coding
	
	print "({0:>6.2f} sec)".format(time.time() - tick)			
	export_output(line_args)	# Export the Inverted File structure to a JSON file

	sys.exit(0)