import pandas as pd
import os
import numpy as np
import preprocessor as p
from gensim.parsing.preprocessing import remove_stopwords
import mysql.connector
from tabulate import tabulate
import re
from mysql.connector import Error
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import preprocess 
#import glove
import setuptools
import sys
import tokenize
from bs4 import BeautifulSoup 
import urllib 
import itertools
import scipy
import nums_from_string
import logging
import smtplib, ssl
from sqlalchemy import create_engine


def importIDS():
	# load in unique ids
	cnn = create_engine('sqlite:///data/processed/IDS.db').connect()		
	commentIDS = pd.read_sql('commentIDS', cnn)
	commentconIDS = pd.read_sql('commentconIDS', cnn)
	commentmedIDS = pd.read_sql('commentmedIDS', cnn)
	cnn.close()
	return commentIDS, commentconIDS, commentmedIDS

fileSet = []
def importCSV():
	try:
		global fileSet
		#open directory
		dict1 = {'fileCommentsContext': os.listdir(os.getcwd() + '\\data\\comments-context'),'fileCommentsMedia': os.listdir('\\data\\comments-media')}
		#grab comments folder
		fileComments = os.listdir('\\data\\comments')#IF FILE NOT IN [CSV FOR PROCESSED FILE NAMES]
		#set it up as the reference folder
		ref = [re.sub(r'\D', '', i) for i in fileComments] #unique identifiers across 6 folders is date and time so we extract numbers only
		counter = 0 #quick fix for init reference 
		for item in ref:#go through every file in the reference folder
			if counter == 0:#pull the index of reference file
				indexRef = ref.index(item)
				fileSet = fileComments[indexRef]#store comments xxxx.csv n
				for number in range(0,len(dict1)):#looping not ideal; force to iterate over 5 dict folders
					try:
						fileDates = [re.sub('\D', '', i) for i in list(dict1.values())[number]]#change files to numbers in other 5 folders
						indexFile = fileDates.index(item)#if a file matches reference save index
						counter += 1
					except:#skip if comments/comments-media/comments-context match doesntexist
						pass
					else:
						fileSet.append(list(dict1.values())[number][indexFile])#store 5 other type xxxx.csv names	
	except Exception as e:
		logging.exception("() error: %s", e)
	return fileSet


def importData():
	try:
		comments = pd.read_csv('comments\\' + fileSet[0])
		commentsContext =  pd.read_csv('comments-context\\' + fileSet[1])
		commentsMedia =  pd.read_csv('comments-media\\' + fileSet[2])
	except Exception as e:
		logging.exception("importData() error: %s", e)
	return comments, commentsContext, commentsMedia

def importSQl():
	try:
		cnx = create_engine('sqlite:///data/sql/comment.db').connect()# table named 'contacts' will be returned as a dataframe
		t = sq.text("""SELECT * 
						FROM comments 
						WHERE id 
						IN :commentIDS; 
		""")
		t = t.bindparams(values=tuple(commentIDS))
		comments = pd.read_sql(t, cnx)

		tc = sq.text("""SELECT * 
						FROM commentsContent
						WHERE id 
						IN :commentconIDS; 
		""")
		tc = tc.bindparams(values=tuple(commentconIDS))
		commentsContext = pd.read_sql(tc, cnx)

		tm = sq.text("""SELECT * 
						FROM commentsMedia 
						WHERE id 
						IN :commentmedIDS; 
		""")
		tm = tm.bindparams(values=tuple(commentmedIDS))
		commentsMedia = pd.read_sql(tm, cnx)
		cnx.close()
	except Exception as e:
			logging.exception("importSQL() error: %s", e)
	return comments, commentsContext, commentsMedia

def transform():
	try:
		url = np.loadtxt('\\data\\url.txt')
		comments.columns = [re.sub(r'\.', "",c) for c in comments.columns]
		commentsContext.columns = [re.sub(r'\.', "",c) for c in commentsContext.columns]
		commentsMedia.columns = [re.sub(r'\.', "",c) for c in commentsMedia.columns]
		#change names and details to join on
		comments.rename(columns={'attachmentsmedia_keys': 'media_key'}, inplace=True)
		comments['media_key'] = comments['media_key'].str.strip('[]').str.strip('\'')
		commentsContext.rename(columns={'tweet_id': 'id'}, inplace=True)
		commentsContext['id'] = commentsContext['id'].astype(str)
		comments.columns = [str(col) + '_comm' for col in comments.columns]#make comment colnames unique
		comments.rename(columns={'id_comm': 'id'}, inplace=True)#but keep merge names
		comments.rename(columns={'media_key_comm': 'media_key'}, inplace=True)
	except Exception as e:
		logging.exception("transform() error: %s", e)
	return comments, commentsContext, commentsMedia

def process():
	try:
		comments['links_comm'] = comments['text_comm'].str.extract(url)#extract any links from text into new column
		comments['text_comm'] = comments['text_comm'].str.replace(url,"").replace(np.nan, "")#remove links from text
		comments['totalwords_comm'] = [len(x.split()) for x in comments['text_comm'].tolist()]#word count to new column
		#REMOVE STOPWORDS
		#lower case
	except Exception as e:
		logging.exception("process() error: %s", e)
	return comments

def completeID():
	try:
		t = comments['id']
		tc = commentsContext['comment_id']
		tm = commentMedia['media_key']
		t.to_sql("commentIDS", sqlite_connection, if_exists='append')
		tc.to_sql("commentconIDS", sqlite_connection, if_exists='append')
		tm.to_sql("commentmedIDS", sqlite_connection, if_exists='append')
		sqlite_connection.close()
	except Exception as e:
		logging.exception("() error: %s", e)
	return

def exportSQL():
	try:
		engine = create_engine('sqlite:///processed/comment.db', echo=True)
		sqlite_connection = engine.connect()
		comments.to_sql("comments", sqlite_connection, if_exists='append')
		commentsContext.to_sql("commentsContext", sqlite_connection, if_exists='append')
		commentsMedia.to_sql("commentsMedia", sqlite_connection, if_exists='append')
		sqlite_connection.close()
	except Exception as e:
		logging.exception("() error: %s", e)

def runall():
    importIDS()
    transform()
    process()
    #sentiment()
    completeID()
    exportSQL()
  
def main(importType):#run processing functions here
	if importType == 'csv':
		importCSV()
		importData()
		runall()
	elif importType == 'sql':
		importSQl()
		runall()

main('csv')
