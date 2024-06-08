import pandas as pd
import os
from pandas.io.sql import read_sql_query
import sqlalchemy as sq
import numpy as np
import preprocessor as p
from gensim.parsing.preprocessing import remove_stopwords
import mysql.connector
from sqlalchemy.sql.sqltypes import BIGINT
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
import textcleaner
import nltk
from textblob import TextBlob
import warnings
import uuid
from datetime import datetime, timedelta
import text2vec
from nltk.corpus import stopwords
from alembic import op
from sqlalchemy import Column, INTEGER, ForeignKey
warnings.filterwarnings("ignore")
import multiprocessing as mp
from sqlalchemy import Column, Integer, MetaData, Table, create_engine, String, BigInteger,Text, update, and_, select, func, types
from sqlalchemy.sql import select
from sqlalchemy.dialects import sqlite
from sqlalchemy.orm import mapper, sessionmaker, Session
import alembic
import itertools

import sqlite3 
stop = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've"]





def getSubjectivity(text):
	return TextBlob(text).sentiment.subjectivity

# Create a function to get the polarity
def getPolarity(text):
	return  TextBlob(text).sentiment.polarity

def getAnalysis(score):
	if score < 0:
		return 'Negative'
	elif score == 0:
		return 'Neutral'
	else:
		return 'Positive'
   

def avg_letter(sentence):
	words = sentence.split()
	if len(words)>0:
		return (sum(len(word) for word in words)/len(words))


def tweets_REP(chunk):
	tweets = chunk
	try:
		#remove periods because they are functional in python
		tweets.columns = [re.sub(r'\.', "",c) for c in tweets.columns]
		tweets.rename(columns={'attachmentsmedia_keys': 'media_key', 'id':'tweet_id', 'public_metricsretweet_count': 'retweet_count','public_metricsreply_count': 'reply_count','public_metricslike_count': 'like_count','public_metricsquote_count':'quote_count'}, inplace=True)#change media keys to match xxx-media
		tweets['text']=tweets['text'].astype(str).fillna('').apply(str)

		tweets['tweet_id'] = tweets['tweet_id'].astype(np.int64)#change to float, the smallest represtnation of the number
		
		tweets['media_key'] = tweets['media_key'].str.strip('[]').str.strip('\'')#remove brackets and quotes
		tweets['links'] = tweets['text'].str.extract(r"(http\S+)",)#extract any links from text into new column
		tweets['text'] = tweets['text'].str.replace(r"(http\S+)","") #remove links fro
		tweets['text'] = [" ".join((str.lower(word)) for word in x.split()) for x in tweets.text]#Change to lowercase
  
		tweets['hashtags'] = tweets['text'].apply(lambda x: re.findall(r'\B#\w*[a-zA-Z]+\w*', x)).astype(str) # new column #s
		tweets['mentions'] = tweets['text'].apply(lambda x: re.findall(r'\B@\w*[a-zA-Z]+\w*', x)).astype(str) #new column @s
		tweets['subjectivity'] = tweets['text'].apply(getSubjectivity) #get subjectivity sentiment  
		tweets['polarity'] = tweets['text'].apply(getPolarity) #subjectivity -> polarity
		tweets['analysis'] = tweets['polarity'].apply(getAnalysis)#polarity -> analysiss
		tweets['avg_polar_convo'] = tweets.groupby('conversation_id')['polarity'].transform('mean') #average polarity
		tweets['avg_sent_convo'] = tweets['avg_polar_convo'].apply(getAnalysis) #average polarity -> average sentiment


		tweets['text'] = [" ".join(re.sub(r"#(\w+)", '',word) for word in x.split()) for x in tweets['text'].tolist()]#remove #s
		tweets['text'] = [" ".join(re.sub(r"@(\w+)", '',word) for word in x.split()) for x in tweets['text'].tolist()]#remove @s
		tweets['text'] = tweets['text'].astype(str).str.replace('[^\w\s]','') #drop punctuation
		tweets['text'] = [" ".join(x for x in x.split() if x not in stop) for x in tweets.text.tolist()] # remove sstopwords'
		tweets['totalwords'] = [len(x.split()) for x in tweets['text'].tolist()]#word count to new column
		tweets['avg_letter'] = tweets['text'].apply(lambda x: avg_letter(x)) #average number of letters in word per tweet
		tweets['avg_words_convo'] =tweets.groupby('conversation_id')['totalwords'].transform('mean')
		tweets['avg_letters_convo'] =tweets.groupby('conversation_id')['avg_letter'].transform('mean')


		#tweets['text'] = [" ".join(str(TextBlob(word).correct()) for word in x.split()) for x in tweets.text.tolist()]#spellcheck --- takes too long

		tweets = tweets.add_suffix('_o')
		#clean = create_engine('sqlite:///C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data/processed/tweets/clean-jan6_orig.db').connect()
		clean = create_engine('sqlite:///C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data_22/processed/jan_clean.db').connect()
		tweets.to_sql("jan-o", clean, if_exists='append', index=False)
		clean.close()
		
		tweetoIDS = pd.Series()
		tweetoIDS = pd.Series(tweets.unique_id_o)
		tweetoIDS = tweetoIDS.rename('tweetoIDS')
		
		log = pd.DataFrame(index = range(len(tweets.unique_id_o)))
		
		log['Stage'] = ''.join('Tweet-Comment ID:')
		log['Tweet ID'] = tweets.tweet_id_o
		log['Row'] = ''.join('Tweets Row ID:')
		
		log['Tweet Row ID'] = tweets.unique_id_o

		logging.info((log.to_string()) )
	except Exception as e:
		logging.exception('tweets error: %s', e)
	return tweetoIDS

def tweets_MED(chunk2):

	try:
		tweetsMedia = chunk2
		tweetsMedia.columns = [re.sub(r'\.', "",c) for c in tweetsMedia.columns]
		tweetsMedia.rename(columns={'public_metricsview_count':'media_view_count'}, inplace=True)
		tweetsMedia = tweetsMedia.add_suffix('_o')

		tweetmedoIDS = pd.Series()
		tweetmedoIDS = pd.Series(tweetsMedia.unique_id_o)
		tweetmedoIDS = tweetmedoIDS.rename('tweetmedoIDS')

		#clean = create_engine('sqlite:///C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data/processed/tweets/clean-jan6_orig.db').connect()
		clean = create_engine('sqlite:///C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data_22/processed/jan_clean.db').connect()

		tweetsMedia.to_sql("jan-media-o",clean, if_exists='append', index=False)
		clean.close()

		log = pd.DataFrame(index = range(len(tweetsMedia.unique_id_o)))
		log['Stage'] = ''.join('Media Row ID:')
		log['Tweet ID'] = tweetsMedia.unique_id_o
		logging.info((log.to_string()) )
	except Exception as e:
		logging.exception('media error: %s', e)
	return tweetmedoIDS



def tweets_CON(chunk3):
	try:
		tweetsContext = chunk3
  
		tweetsContext.columns = [re.sub(r'\.', "",c) for c in tweetsContext.columns]
		tweetsContext.rename(columns = {'domainid':'domain_id','domainname':'domain_name','entityid':'entity_id','entityname':'entity_name'}, inplace = True)
		tweetsContext=tweetsContext.add_suffix('_o')

		tweetconoIDS = pd.Series()
		tweetconoIDS = pd.Series(tweetsContext.unique_id_o)
		tweetconoIDS = tweetconoIDS.rename('tweetconoIDS')

		#clean = create_engine('sqlite:///C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data//processed/tweets/clean-jan6_orig.db').connect()
		clean = create_engine('sqlite:///C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data_22/processed/jan_clean.db').connect()

		tweetsContext.to_sql("jan-context-o", clean, if_exists='append', index=False)
		clean.close()

   
		log = pd.DataFrame(index = range(len(tweetsContext.unique_id_o)))
		log['Stage'] = ''.join('Context Row ID:')
		log['Tweet ID'] = tweetsContext.unique_id_o
		logging.info((log.to_string()) )
  
	except Exception as e:
		logging.exception('context error: %s', e)
	return tweetconoIDS
 
def main():#run processing functions here
	logging.basicConfig(filename='C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data_22/processed/logs/' + 'TWEET' +datetime.today().strftime("%m-%d-%Y-%H-%M-%S") + '.log',level=logging.INFO, format='%(message)s', force=True)
	logging.info('Start time: %s', datetime.now())
	
	chunkTW= 10000
	chunkTWM = 10000
	chunkTWC = 10000
	#cnx = create_engine('sqlite:///C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data/sql/tweets/jan6_orig.db').connect()
	cnx = create_engine('sqlite:///C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data_22/jan_dirty.db').connect()

	for chunk in pd.read_sql(sql =" SELECT * FROM 'tweets-o' WHERE unique_id not in (SELECT tweetoIDS FROM tweetoIDS)", con = cnx, chunksize = chunkTW):
		a = tweets_REP(chunk)
		a.to_sql('tweetoIDS',con=cnx, index=False, if_exists='append')


	for chunk2 in pd.read_sql(sql =" SELECT * FROM 'tweets-media-o' WHERE unique_id not in (SELECT tweetmedoIDS FROM tweetmedoIDS)", con = cnx, chunksize = chunkTWM):
		b = tweets_MED(chunk2)
		b.to_sql('tweetmedoIDS', con=cnx, index = False, if_exists='append')
  
	for chunk3 in pd.read_sql(sql =" SELECT * FROM 'tweets-context-o' WHERE unique_id not in (SELECT tweetconoIDS FROM tweetconoIDS)", con = cnx, chunksize = chunkTWC):
		c = tweets_CON(chunk3)
		c.to_sql('tweetconoIDS', con = cnx, index = False, if_exists = 'append')
  
	cnx.close()
	logging.info('End time: %s', datetime.now())
	

main()
 


quit()




