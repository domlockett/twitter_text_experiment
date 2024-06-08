

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
from sklearn.feature_extraction.text_o import CountVectorizer
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
from sqlalchemy import INTEGER, create_engine
import text_ocleaner
import nltk
from text_oblob import text_oBlob
import warnings
import uuid
from datetime import datetime, timedelta
import text_o2vec
from nltk.corpus import stopwords
warnings.filterwarnings("ignore")
from sqlalchemy import tuple_
from sqlalchemy import Column, Integer, MetaData, Table, create_engine, String, BigInteger,Text, update, and_, select, func, types,Float
from sqlalchemy.dialects import sqlite
from sqlalchemy.orm import mapper, sessionmaker, Session
import alembic.migration
import alembic.operations
BigIntegerType = BigInteger()
BigIntegerType = BigIntegerType.with_variant(sqlite.INTEGER(), 'sqlite')
from sqlalchemy import create_engine, inspect
from sqlalchemy_utils.functions import database_exists, create_database
cnx = create_engine('sqlite:///C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data_22/processed/jan_clean.db').connect()

conn = create_engine('sqlite:///C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data_22/joan_dirty.db').connect()
gett =inspect(cnx)
gett.get_table_names()
pd.read_sql(sql =" SELECT * FROM 'tweets-o'", con = conn)

# ctx = alembic.migration.MigrationContext.configure(conn)
# op = alembic.operations.Operations(ctx)

#pd.Series(0.0, name='tweetIDS').to_sql('tweetIDS', con=conn, index=False)
# pd.Series(0.0, name='tweetmedoIDS').to_sql(#'tweetmedoIDS', con=conn, index=False) 
# pd.Series(0.0, name = 'tweetconoIDS').to_sql('tweetconoIDS', con=conn, index=False)

# conn.execute("DROP TABLE 'tweetconoIDS'")
# conn.execute("DROP TABLE 'tweetmedoIDS'")
#conn.execute("DROP TABLE 'tweetIDS'")
