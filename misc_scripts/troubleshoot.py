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
from alembic import op
import multiprocessing as mp
from sqlalchemy import Column, Integer, MetaData, Table, create_engine
from sqlalchemy.orm import mapper, sessionmaker, Session
import time
warnings.filterwarnings("ignore")



a = [0, 1, 2, 3, 4, 5]
b =[0, 10, 20, 30, 40, 50]
# here is your data, in two numpy arrays
file = open("list.txt", "w")
for index in range(len(a)):
    file.write(str(a[index]) + " " + str(b[index]) + "\n")
file.close()

#split process bones
print("Number of processors: ", mp.cpu_count())
pool = mp.Pool(mp.cpu_count())
# results = [pool.apply(function=z, args=y) for row in data]
pool.close()

# import ids
cnx = create_engine('sqlite:///data/processed/clean_c.db').connect()
ids3 = pd.read_sql_table('Ids', cnx )
tweets3 = pd.read_sql_table('tweets', cnx )
tweetsMedia3= pd.read_sql_table('tweets-media', cnx )
tweetsContext3= pd.read_sql_table('tweets-context', cnx )

cnx.close()
ids3.tweetIDS =0
#import clean data
cnx = create_engine('sqlite:///data/sql/tweet.db').connect()
 # table named 'contacts' will be returned as a dataframe.
tweets2 = pd.read_sql_table('tweets', cnx )
tweetsMedia2 = pd.read_sql_table('tweets-media', cnx)
tweetsContext2 = pd.read_sql_table('tweets-context', cnx )
cnx.close()

cnx = create_engine('sqlite:///data/sql/tweet1.db').connect()
 # table named 'contacts' will be returned as a dataframe.
tweets = pd.read_sql_table('tweets', cnx )
tweetsMedia = pd.read_sql_table('tweets-media', cnx)
tweetsContext = pd.read_sql_table('tweets-context', cnx )
cnx.close()


tweets2.columns.tolist()[13] =='unique_id'
tweetsMedia2.columns.tolist()[5]=='unique_id'
tweetsContext2.columns.tolist()[5] == 'unique_id'
#make subset
tw = tweets[:1000]
twM = tweetsMedia[:1000]
twC = tweetsContext[:1000]

# #write ID subset
# cnx = create_engine('sqlite:///data/sql/clean tests/tweet1.db').connect()
# tw.to_sql("tweetIDS",cnx, if_exists='append', index=False)
# twM.to_sql("tweetconIDS",cnx, if_exists='append', index=False)
# twC.to_sql("tweetmedIDS",cnx, if_exists='append', index=False)
# cnx.close()

#write TWEET subset
cnx = create_engine('sqlite:///data/processed/clean_c.db').connect()
ids3.to_sql("Ids",cnx, index=False)
tweets3.to_sql("tweets",cnx, index=False)
tweetsMedia3.to_sql('tweets-media', cnx , index=False)
tweetsContext3.to_sql('tweets-context', cnx , index=False)
cnx.close()

#write TWEET subset
cnx = create_engine('sqlite:///data/sql/tweet2.db').connect()
tw.to_sql("tweets",cnx, if_exists='append', index=False)
twM.to_sql('tweets-media', cnx , index=False)
twC.to_sql('tweets-context', cnx , index=False)
cnx.close()


#write TWEET subset
cnx = create_engine('sqlite:///data/sql/tweet3.db').connect()
tw.to_sql("tweets",cnx, if_exists='append', index=False)
twM.to_sql('tweets-media', cnx , index=False)
twC.to_sql('tweets-context', cnx , index=False)
cnx.close()

ids3.tweetIDS =0

#write TWEET subset
cnx = create_engine('sqlite:///data/processed/clean_c.db').connect()
ids3.to_sql("Ids",cnx, if_exists='append', index=False)

cnx.close()


#write TWEET subset
cnx = create_engine('sqlite:///data/sql/tweet5.db').connect()
tw.to_sql("tweets",cnx, if_exists='append', index=False)
twM.to_sql('tweets-media', cnx , index=False)
twC.to_sql('tweets-context', cnx , index=False)
cnx.close()


#write TWEET subset
cnx = create_engine('sqlite:///data/sql/tweet6.db').connect()
tw.to_sql("tweets",cnx, if_exists='append', index=False)
twM.to_sql('tweets-media', cnx , index=False)
twC.to_sql('tweets-context', cnx , index=False)
cnx.close()


#write TWEET subset
cnx = create_engine('sqlite:///data/sql/tweet7.db').connect()
tw.to_sql("tweets",cnx, if_exists='append', index=False)
twM.to_sql('tweets-media', cnx , index=False)
twC.to_sql('tweets-context', cnx , index=False)
cnx.close()


#write TWEET subset
cnx = create_engine('sqlite:///data/sql/tweet8.db').connect()
tw.to_sql("tweets",cnx, if_exists='append', index=False)
twM.to_sql('tweets-media', cnx , index=False)
twC.to_sql('tweets-context', cnx , index=False)
cnx.close()


#write TWEET subset
cnx = create_engine('sqlite:///data/sql/tweet9.db').connect()
tw.to_sql("tweets",cnx, if_exists='append', index=False)
twM.to_sql('tweets-media', cnx , index=False)
twC.to_sql('tweets-context', cnx , index=False)
cnx.close()


# cnx = create_engine('sqlite:///data/sql/tweet.db').connect()
# # # # table named 'contacts' will be returned as a dataframe.
# tw.to_sql("tweets",cnx, if_exists='append', index=False)
# twM.to_sql('tweets-media', cnx , index=False)
# twC.to_sql('tweets-context', cnx , index=False)
# cnx.close()
