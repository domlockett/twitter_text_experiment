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
from sqlalchemy import create_engine, Table, Column, Integer, Unicode, MetaData, String, Text, update, and_, select, func, types
import alembic.migration
import alembic.operations
warnings.filterwarnings("ignore")



proj = 'sqlite:///C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data/'


srcEngine = create_engine(proj + 'processed/tweets/clean_orig.db') # change this for your source database
srcEngine._metadata = MetaData(bind=srcEngine)
srcEngine._metadata.reflect(srcEngine) #  get columns from existing table
srcTable1 = Table('tweets-o', srcEngine._metadata)
srcTable2 = Table('tweets-media-o', srcEngine._metadata)
srcTable3 = Table('tweets-context-o', srcEngine._metadata)


# create engine and table object for newTable
destEngine = create_engine(proj + 'processed/comments/clean_comm_dl.db') # change this for your destination database
destEngine._metadata = MetaData(bind=destEngine)
destTable1 = Table('tweets-o', destEngine._metadata)
destTable2 = Table('tweets-media-o', destEngine._metadata)
destTable3 = Table('tweets-context-o', destEngine._metadata)




# engine =  create_engine('sqlite:///C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data/processed/comments/clean_comm_dl.db') # connection properties stored


# metadata = MetaData() # stores the 'production' database's metadata
# destTable1 = Table('tweets-o',metadata)
# destTable2 = Table('tweets-media-o', metadata)
# destTable3 = Table('tweets-context-o', metadata)


# destTable1.drop(engine) 
# destTable2.drop(engine) 
# destTable3.drop(engine) 


# copy schema and create newTable from oldTable
for column in srcTable1.columns:
    destTable1.append_column(column.copy())
destTable1.create()

for column in srcTable2.columns:
    destTable2.append_column(column.copy())
destTable2.create()

for column in srcTable3.columns:
    destTable3.append_column(column.copy())
destTable3.create()


cnx1 = create_engine('sqlite:///C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data/processed/tweets/clean_orig.db').connect()
cnx2 = create_engine('sqlite:///C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data/processed/comments/clean_comm_dl.db').connect()



for chunk in pd.read_sql(sql =" SELECT * FROM 'tweets-o'", con = cnx1, chunksize = 10000):
	chunk.to_sql('tweets-o',con=cnx2, index=False, if_exists='append')


for chunk2 in pd.read_sql(sql =" SELECT * FROM 'tweets-media-o'", con = cnx1, chunksize = 10000):
	chunk2.to_sql('tweets-media-o', con=cnx2, index = False, if_exists='append')

for chunk3 in pd.read_sql(sql =" SELECT * FROM 'tweets-context-o'", con = cnx1, chunksize = 10000):
	chunk3.to_sql('tweets-context-o', con = cnx2, index = False, if_exists = 'append')

cnx1.close()
cnx2.close()

