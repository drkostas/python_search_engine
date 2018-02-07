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

inverted_file = {}							# The Inverted File data structure

def set_argParser():
	""" The build_index script's arguments presentation method."""
	argParser = argparse.ArgumentParser(description="Script's objective is to query the Inverted File constructed previously after executing BuildIndex script.")
	argParser.add_argument('-I', '--input_file', type=str, default=os.path.dirname(os.path.realpath(__file__)) + os.sep + 'inverted_file.txt', help='The file path of the Inverted File constructed from BuildIndex. Default:' + os.path.dirname(os.path.realpath(__file__)) + os.sep + 'inverted_file.txt')
	return argParser

def check_arguments(argParser):
	""" Parse and check the inserted command line args."""
	return argParser.parse_args()

def retrieve_inverted_index(line_args):
	""" Retrieve the Inverted Index."""
	global inverted_file		# The Inverted File data structure
	with open(line_args.input_file, 'r') as fh:
		inverted_file = json.load(fh)

if (__name__ == "__main__") :
	argParser = set_argParser()				# The argument parser instance
	line_args = check_arguments(argParser)	# Check and redefine, if necessary, the given line arguments 
	retrieve_inverted_index(line_args)
	counter = 0
	sumdis = 0
	sumnum = 0
	minn = 10000
	maxx = 0
	for k,word in inverted_file.iteritems():
		for key,doc in word['l'].iteritems():
			appearence = 0
			if doc[1] !=0 and doc[1] < minn:
				minn = doc[1]
			if doc[1] >maxx:
				maxx = doc[1]
			for appear in doc[0]:
				appearence += 1
				sumnum = sumnum + appear
				if appearence >1:					
					counter += 1
					sumdis += appear
	print "-----------------------------------------------------------------------------------------"
	print "The mean distance between the appearences of each word is: ",round(float(sumdis)/counter)
	print
	print "The mean location(location or distance) of each word is: ",round(float(sumnum)/counter)
	print
	print "The maximum tf*idf is ",maxx
	print
	print "The minimum tf*idf(except zero) is ",minn
	print "-----------------------------------------------------------------------------------------"
	sys.exit(0)