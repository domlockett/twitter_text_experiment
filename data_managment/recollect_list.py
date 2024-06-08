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

#original ids, then cleaned ids collect whats missing
db_destination = create_engine('sqlite:///C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data_22/processed/convo_ids.db').connect()
#db_all = create_engine('sqlite:///C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data_22/processed/clean_all.db').connect()


#grab original ids put into dataframe

# adding column name to the respective columns
#original_ids = pd.DataFrame(columns = ['orig_ids'])
#original_ids['orig_ids'] = pd.read_fwf('C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/conversation_ids04-23-21.txt')
#original_ids.to_sql('original_ids',con=db_destination, index=False, if_exists='append')

#gram the completed ids put into dataframe
#complete_ids = pd.DataFrame(columns = ['done_ids1'])


##complete_ids['done_ids2'] = pd.read_sql(sql =" SELECT conversation_id FROM tweets", con = db_all)
###complete_ids.done_ids.to_sql('complete_ids',con=db_destination, index=False, if_exists='append')



#new_list.to_sql('recollection',con=db_destination,index=False, if_exists='append')


#pd.read_sql(" SELECT o.orig_ids FROM 'original_ids' o WHERE NOT EXISTS (SELECT 1 FROM 'complete_ids' c WHERE c.done_ids =o.orig_ids)",con=db_destination).to_csv('C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data_22/processed/conversation_ids03-22.txt', header=True, sep='\n', index=False, encoding='utf-8')


# = pd.read_sql("SELECT l.orig_ids FROM 'original_ids' l LEFT JOIN 'complete_ids' r ON r.done_ids = l.orig_ids WHERE r.done_ids IS NULL",con=db_destination)

#.to_csv('C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data_22/processed/conversation_ids03-22.txt', header=True, sep='\n', index=False, encoding='utf-8')





test = pd.read_sql("SELECT t1.orig_ids FROM 'original_ids' t1 LEFT OUTER JOIN 'complete_ids' t2 ON t1.orig_ids = cast(t2.done_ids, float) WHERE cast(t2.done_ids,float) IS NULL",con=db_destination)

test.to_sql('recollect.db', con = db_destination)

#db_all.close()
orig = pd.read_sql(" SELECT orig_ids FROM 'original_ids'",con=db_destination)


done = pd.read_sql("SELECT done_ids FROM 'complete_ids'",con=db_destination)
done.done_ids = done.done_ids.astype(float)

recollect = orig.orig_ids[~orig.orig_ids.isin(done.done_ids)]
recollect.to_csv('C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data_22/processed/conversation_ids03-22.txt', header=True, sep='\n', index=False, encoding='utf-8')











db_destination.close()
