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
import multiprocessing as mp
from sqlalchemy import Column, Integer, MetaData, Table, create_engine, String, BigInteger,Text, update, and_, select, func, types,Float
from sqlalchemy.dialects import sqlite
from sqlalchemy.orm import mapper, sessionmaker, Session
import alembic.migration
import alembic.operations
BigIntegerType = BigInteger()
BigIntegerType = BigIntegerType.with_variant(sqlite.INTEGER(), 'sqlite')

import alembic
def unique_id():
	conn = create_engine('sqlite:///C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data_22/jan_dirty.db').connect()
	ctx = alembic.migration.MigrationContext.configure(conn)
	op = alembic.operations.Operations(ctx)
	pd.Series(0.0, name='tweetoIDS').to_sql('tweetoIDS', con=conn, index=False)
                   
	pd.Series(0.0, name='tweetmedoIDS').to_sql('tweetmedoIDS', con=conn, index=False) 
	pd.Series(0.0, name = 'tweetconoIDS').to_sql('tweetconoIDS', con=conn, index=False)
	with op.batch_alter_table("tweets-o", recreate= "always" ) as batch_op:
		batch_op.alter_column( column_name= 'id', type_= Integer)
  
	with op.batch_alter_table("tweets-o", recreate= "always" ) as batch_op:
		batch_op.add_column(Column('unique_id', BigIntegerType, autoincrement=True))
		batch_op.create_primary_key("pk_tweets", ["unique_id"])

	with op.batch_alter_table("tweets-media-o", recreate= "always" ) as batch_op:
		batch_op.add_column(Column('unique_id', BigIntegerType, autoincrement=True))
		batch_op.create_primary_key("pk_tweets_media", ["unique_id"])
  
	with op.batch_alter_table("tweets-context-o", recreate= "always" ) as batch_op:
		batch_op.add_column(Column('unique_id', BigIntegerType, autoincrement=True))
		batch_op.create_primary_key("tweets-context", ["unique_id"])
	conn.close()
unique_id()
