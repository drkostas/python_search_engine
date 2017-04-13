#!C:\Program Files\Python35\python.exe
# Querying the Inverted Index: There are 2 types of queries we want to handle:
#   - Standard queries: where at least one of the words in the query appears in the document,
#   - Phrase queries: where all the words in the query appear in the document in the same order. 
# http://aakashjapi.com/fuckin-search-engines-how-do-they-work/

import os
import re
import sys
import json
import time
import argparse
import cgi  
import nltk

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from functools import reduce

form = cgi.FieldStorage()
print("Content-type: text/html\n\n")
inverted_file = []							# The Inverted File data structure
docnames = []								# The Document names list
indeces = []								# The indeces of the query lemmas in the inverted file
wp_tokenizer = WordPunctTokenizer()			# Tokenizer instance
wnl_lemmatizer = WordNetLemmatizer()		# Wordnet Lemmatizer instance
stop_words = stopwords.words('english')		# English stop words list



def set_argParser():
	""" The build_index script's arguments presentation method."""
	argParser = argparse.ArgumentParser(description="Script's objective is to query the Inverted File constructed previously after executing BuildIndex script.")
	argParser.add_argument('-I', '--input_file', type=str, default=os.path.dirname(os.path.realpath(__file__)) + os.sep + 'inverted_file.txt', help='The file path of the Inverted File constructed from BuildIndex. Default:' + os.path.dirname(os.path.realpath(__file__)) + os.sep + 'inverted_file.txt')
	return argParser



def check_arguments(argParser):
	""" Parse and check the inserted command line args."""
	return argParser.parse_args()



def retrieve_files(line_args):
	""" Retrieve the Inverted Index."""
	global inverted_file		# The Inverted File data structure
	global docnames				# The Document names data structure
	with open(line_args.input_file, 'r') as fh:
		# Inverted File Form for the words find,finished,finland and go:
		# [ 
		# 	[ "find",0,0,{"document  number":[ [ positions in doc ], tf*idf ]} ],
		# 	[ "ished",3,1,{"document  number":[ [ positions in doc ], tf*idf ]} ],
		# 	[ "land",3,2,{"document  number":[ [ positions in doc ], tf*idf ]}],
		# 	[ "go",0,3,{"document  number":[ [ positions in doc ], tf*idf ]}] 
		# ]
		inverted_file = json.load(fh)

	with open('docnames.txt', 'r') as fh:
		docnames = json.load(fh)



def binarySearch(item):

    first = 0
    last = len(inverted_file)-1
    found = False
    while first<=last and not found:    
        midpoint = (first + last)//2
        current = inverted_file[midpoint][:]
        current_whole = current[:]
	
        if current[2]!=0:
        	current_whole[0] = inverted_file[midpoint-current[2]][0][0:current[1]]+current[0]
        if current_whole[0] == item:
            found = True
            ind = inverted_file.index(current)
        else:
            if item < current_whole[0]:
                last = midpoint-1
            else:
                first = midpoint+1
    if found:
    	return ind
    else:
    	return -1

def ranking_tfidf(func):
	""" Ranking the queries regarding tf*idf."""
	def print_top_10(query_lemmas):
		""" Print the top 10 answers regarding the sum of tf*idf involved query lemmas."""
		retrieved_documents = func(query_lemmas)

		# Exit if the given list of documents for retrieving is empty. 
		if (not(retrieved_documents)):
			return 0

		# For each document in the retrieving list calculate the respective sum of tf*idf of the individual lemma.		
		retrieved_documents_tfidf = {docid: reduce(lambda x, y: x + y, [inverted_file[index][3][str(docnames.index(docid)+1)][1] if str(docnames.index(docid)+1) in inverted_file[index][3] else 0 for index in indeces]) for docid in retrieved_documents}
		
		# Print the descending ordered list of the retrieving documents regarding the previously calculated sum of tf*idf score.
		tf_idf_sorted = sorted(list(retrieved_documents_tfidf.keys()), key=lambda x: -retrieved_documents_tfidf[x])
		print(" ")
		print("{0:>2} {1:>20} {2:>10}".format("#", "Document Name", "tf*idf")) 
		for i in range(len(tf_idf_sorted[:10])):
			print("{0:>2} {1:>20} {2:>10}".format(i + 1, tf_idf_sorted[i], retrieved_documents_tfidf[tf_idf_sorted[i]]))
		print(" ")
		print(" ")
		print(" ")

	return print_top_10



@ranking_tfidf
def standard_query(query_lemmas):
	""" Standard query application:
	After sanitizing/wrangling the input query we retrieve the inverted list of the remaining terms/lemmas and which we aggregate and union them.
	"""
	global inverted_file, docnames, indeces

	documents = [docnames[int(docid)-1] for index in indeces for docid in inverted_file[index][3]]	
	standard_query_docs = list(set(documents))
	print(" Standard Querying:", "No relevant document!" if (len(standard_query_docs) < 1) else " ".join(["{0:>4}".format(len(standard_query_docs)), "Documents :", ",".join(standard_query_docs)]))

	return standard_query_docs



@ranking_tfidf
def phrase_query(query_lemmas):
	"""Phrase query appication
	After sanitizing/wrangling the input query we run a single word query for every lemma found and add each of these of results to our total list. 'common_documents' is the setted list that contains all the documents that contain all the words in the query.
	Then we check them for ordering. So, for every list in the intermediate results, we first make a list of lists of the positions of each wordd in the input query. Then we use two nested for loops to iterate through this list of lists. If the words are in the proper order, 
	"""
	global inverted_file,docnames,indeces

	for i in range(0, len(query_lemmas)):
		common_documents = set([docid for docid in inverted_file[indeces[i]][3]]) if (i == 0) else common_documents.intersection(set([docid for docid in inverted_file[indeces[i]][3]]))
		# It doesn't take the document names but the numbers instead
		if (len(common_documents) == 0):
			break

	if (len(common_documents) < 1):
		print(" Phrase Querying:", "No relevant document!")
		return []

	phrase_query_docs = []
	for docid in list(common_documents):
		# Index the query lemmas
		# query_lemmas: project gutenberg archive foundation
		# init_zipped : [('project', 0), ('gutenberg', 1), ('archive', 2), ('foundation', 3)]
		init_zipped = list(zip(indeces, list(range(len(indeces)))))

		# Find the lemma with the biggest tf*idf value in this document in order to check according to this.
		min_zip = init_zipped[0]
		for i in range(1, len(indeces)):
			if (inverted_file[min_zip[0]][3][docid][1] < inverted_file[indeces[i]][3][docid][1]):
				min_zip = init_zipped[i]
		# Replace the relevant position of the lemmas regarding the least appearances lemma's position.
		# Considering that the lemma 'archive' has the least appearances in this document.
		# rel_min_zipped: [('project', -2), ('gutenberg', -1), ('archive', 0), ('foundation', 1)]
		rel_min_zipped = list(zip(indeces, [i - min_zip[1] for i in range(len(indeces))]))
		positionlist = inverted_file[min_zip[0]][3][docid][0]
		for counter in range(len(positionlist)):
			pos = sum(positionlist[:(counter+1)])
			# Considering that 'archive' term is found in position 91.
			# lemmas           : project gutenberg archive foundation
			# relevant position:     -2      -1       0       1
			# actual position  :     89      90      91      92
			# pos_zipped : [('project', 89), ('gutenberg', 1), ('archive', 1), ('foundation', 1)]
			pos_zipped = list(zip(indeces, [pos + rel_min_zipped[i][1] for i in range(len(rel_min_zipped))]))

			# Foreach query's lemma, if the lemma is found in the calculated position we mark it with '1' otherwise the relevant position is set to '0'
			# If all the checked lemmas, found in the correct calculated positions => This document contain the under checking sequence of terms => Should be retrieved as a valid answer
			if (reduce(lambda x, y: x + y, [1 if (pos_zipped[i][1] in inverted_file[pos_zipped[i][0]][3][docid][0]) else 0 for i in range(len(pos_zipped))]) == len(pos_zipped)):
				phrase_query_docs.append(docnames[int(docid)-1])
				break

	print(" Phrase Querying:", "No relevant document!" if (len(phrase_query_docs) < 1) else " ".join(["{0:>4}".format(len(phrase_query_docs)), "Documents :", ",".join(phrase_query_docs)]))

	return phrase_query_docs



# ---------------------------------------------- #
# Examples of queries for experimenting :
# ---------------------------------------
# full of joy
# full of joy and wisdom
# copyright laws
# Start Of Project Gutenberg
# Project Gutenberg
# Project Gutenberg Literacy Archive Foundation
# Please do not remove this
# Himalayans journals
# ---------------------------------------------- #



if (__name__ == "__main__") :
	argParser = set_argParser()				# The argument parser instance
	line_args = check_arguments(argParser)	# Check and redefine, if necessary, the given line arguments 
	retrieve_files(line_args)				# Retrieve the inverted index, the document names and the lexicon

	
	# -------------------------------------- User Interaction -------------------------------------- #

	query = form["Query"].value
	tick = time.time()
	# List of valid lemmas included in current query
	# query        : Project Gutenberg Literacy Archive Foundation
	# query_lemmas : project gutenberg archive foundation
	query_lemmas = []
	indeces = []

	for word, pos in pos_tag(wp_tokenizer.tokenize(query.lower().strip())):
		# It is proper to sanitize the query like we sanitized the documents documents when we built the index by stemming all the words, 
		# making everything lowercase, removing punctuation and apply the analysis applied while building the index.
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
			continue

		pos = 'v' if (pos.startswith('VB')) else 'n'	# If current term's appearance is verb related then the POS lemmatizer should be verb ('v'), otherwise ('n')
		index = binarySearch(word)
		if (index != -1):
			indeces.append(index)			
			query_lemma_word = wnl_lemmatizer.lemmatize(word, pos) 		# Stemming/Lemmatization				
			query_lemmas.append(query_lemma_word)
					


	if (len(query_lemmas) < 1):
		print("Querying: No relevant document!")
	else:
		# Standard query: After sanitizing/wrangling the input query we retrieve the inverted list of the remaining terms/lemmas and which we aggregate and union them.
		standard_query(query_lemmas)

		# Phrase query: After sanitizing/wrangling the input query we run a single word query for every lemma found and add each of these of results to our total list. 
		# We 'common_documents' the setted list that contains all the documents that contain all the words in the query.
		# Then we check them for ordering. So, for every list in the intermediate results, we first make a list of lists of the positions of each wordd in the input query. 
		# Then we use two nested for loops to iterate through this list of lists. If the words are in the proper order, 
		phrase_query(query_lemmas)
	print("({0:>7.5f} sec)".format(time.time() - tick))
	sys.exit(0)
