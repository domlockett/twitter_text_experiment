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
import alembic
warnings.filterwarnings("ignore")
import alembic.migration
import alembic.operations

cnx = create_engine('sqlite:///data/sql/tweet1.db').connect()
tweets = pd.read_sql_table('tweets', cnx )
tweetsMedia = pd.read_sql_table('tweets-media', cnx)
tweetsContext = pd.read_sql_table('tweets-context', cnx )
cnx.close()

tweets=tweets.drop('unique_id',1)
tweetsMedia=tweetsMedia.drop('unique_id',1)
tweetsContext=tweetsContext.drop('unique_id',1)

twd = tweets.drop_duplicates()
twm = tweetsMedia.drop_duplicates()
twc = tweetsContext.drop_duplicates()

cnx = create_engine('sqlite:///data/sql/tweet3.db').connect()
twd.to_sql('tweets', con = cnx, index = False, if_exists = 'append',chunksize=10000)   
twm.to_sql('tweets-media', con = cnx, index = False, if_exists = 'append',chunksize=10000)
twc.to_sql('tweets-context', con = cnx, index = False, if_exists = 'append',chunksize=10000)
cnx.close()

twd.merge(on=list('id','tweet_id'))