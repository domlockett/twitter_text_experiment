import pandas as pd
import os
import sqlalchemy as sq
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
import textcleaner
import nltk
from textblob import TextBlob
import warnings
import uuid
from datetime import datetime, timedelta
import text2vec
from nltk.corpus import stopwords

warnings.filterwarnings("ignore")




files = [os.listdir(os.getcwd() + '\\original_tweet\\jan6_tweet-csv')]#grab all the files from two other tweet folders
files2 = [os.listdir(os.getcwd() + '\\original_tweet\\jan6_comments-csv')]#grab all the files from two other tweet folders


cnx = create_engine('sqlite:///data_22/jan_dirty.db').connect()

tw = files[0][0:2]
tc = files[0][2:4]
tm = files[0][4:6]


for file in [0,1]:
	for chunk1 in  pd.read_csv(os.getcwd() + '\\original_tweet\\jan6_tweet-csv\\'+tw[1], chunksize = 10000):
		chunk1.to_sql('tweets-o',con=cnx, index=False, if_exists='append')
	for chunk2 in  pd.read_csv(os.getcwd() + '\\original_tweet\\jan6_tweet-csv\\'+tc[1], chunksize = 10000):
		chunk2.to_sql('tweets-context-o',con=cnx, index=False, if_exists='append')
	for chunk3 in  pd.read_csv(os.getcwd() + '\\original_tweet\\jan6_tweet-csv\\'+tm[1], chunksize = 10000):
		chunk3.to_sql('tweets-media-o',con=cnx, index=False, if_exists='append')

	

cnx.close()




tw = files2[0][0:3]
tc = files2[0][3:6]
tm = files2[0][6:9]

cnx = create_engine('sqlite:///data_22/jan_dirty.db').connect()

for file in range(0,3):
	for chunk1 in pd.read_csv(os.getcwd() + '\\original_tweet\\jan6_comments-csv\\'+tw[file], chunksize = 10000):
		chunk1.to_sql('tweets',con=cnx, index=False, if_exists='append')
	for chunk2 in  pd.read_csv(os.getcwd() + '\\original_tweet\\jan6_comments-csv\\'+tm[file], chunksize = 10000):
		chunk2.to_sql('tweets-media',con=cnx, index=False, if_exists='append')

	for chunk3 in  pd.read_csv(os.getcwd() + '\\original_tweet\\jan6_comments-csv\\'+tc[file], chunksize = 10000):
		chunk3.to_sql('tweets-context',con=cnx, index=False, if_exists='append')

cnx.close()

cnx = create_engine('sqlite:///C:/Users/Path/to/Data/data_22/jan_dirty.db').connect()

tweets1 = pd.read_sql_table('tweets', cnx )
tweetsMedia1 = pd.read_sql_table('tweets-media', cnx)
tweetsContext1 = pd.read_sql_table('tweets-context', cnx )
cnx.close()

cnx = create_engine('sqlite:///C:/Users/Path/to/Data/data_22/jan_dirty.db').connect()

tweets = pd.read_sql_table('tweets-o', cnx )
tweetsMedia = pd.read_sql_table('tweets-media-o', cnx)
tweetsContext = pd.read_sql_table('tweets-context-o', cnx )
cnx.close()

tweets1[tweets1.conversation_id == 1347101738785005569]
tweets[tweets.conversation_id == 1347101738785005569]