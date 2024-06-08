from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from pickle import TRUE
import pandas as pd
import time
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
from sqlalchemy import INTEGER, create_engine
import textcleaner
import nltk
from textblob import textBlob
import warnings
import uuid
from datetime import datetime, timedelta
import text2vec
from nltk.corpus import stopwords
warnings.filterwarnings("ignore")
from sqlalchemy import tuple_
from sqlalchemy import Column, Integer, MetaData, Table, create_engine, String, BigInteger,Text, update, and_, select, func, types,Float
from sqlalchemy.dialects import sqlite
from sqlalchemy.orm import mapper, sessionmaker, Session
import alembic.migration
import alembic.operations
import matplotlib.pyplot as plt
import seaborn as sb

#40000

BigIntegerType = BigInteger()
BigIntegerType = BigIntegerType.with_variant(sqlite.INTEGER(), 'sqlite')
from sqlalchemy import create_engine, inspect
cnx = create_engine('sqlite:///C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data_22/processed/clean_all.db').connect()
conn = create_engine('sqlite:///C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data_22/processed/jan_clean.db').connect()


#DESCRIBE dataset
n_estimater = cnx.execute("SELECT COUNT(source) FROM 'tweets'").fetchone()[0]
#n_estimater-129594040 <- estimated 100000000... cool
n_estimateo = cnx.execute("SELECT COUNT(totalwords_o) FROM 'tweets-o'").fetchone()[0]
#n_estimateo -5419606


max_estimater = cnx.execute("SELECT MAX(totalwords) FROM 'tweets'").fetchone()[0]
#max_estimater-140 <- nonsense tweet
max_estimateo = cnx.execute("SELECT MAX(totalwords_o) FROM 'tweets-o'").fetchone()[0]
#max_estimateo-88 


#having 0 minimum words works as long as those with zeros also have links or media associated with them

min_estimater = cnx.execute("SELECT MIN(totalwords) FROM 'tweets'").fetchone()[0]
#min_estimater-0 
min_estimateo = cnx.execute("SELECT MIN(totalwords_o) FROM 'tweets-o'").fetchone()[0]
#min_estimateo-0
#->checking<-

min_details = pd.read_sql("SELECT * FROM 'tweets-o' WHERE LENGTH(text_o) <1", con = cnx)
min_details.links_o# looks like ~200 tweets include no text and  no link

min_details["links_o"] =['None' if v is None else v for v in min_details.links_o]
new_df =  min_details[~min_details["links_o"].str.contains("t.co")]



[min_details[x] for x in new_df["links_o"].index.to_list() ]
min_details[152]









###### LENGTH OF TWEETS ADDITION ######
# Deliver: 
# # Variable added to the dataset
# # Report of the average difference in word count across the entire dataset -- Tweets are 3.11 words longer than their replies
# # Report of the average difference in word count for just the Jan 6 case study

###whole dataset
reply_avg = cnx.execute("SELECT AVG(totalwords) FROM 'tweets'").fetchone()[0]

#reply_avg-7.99
orig_avg = cnx.execute("SELECT AVG(totalwords_o) FROM 'tweets-o'").fetchone()[0]
#orig_avg--11.12


###jan 6 case
reply_avgj = conn.execute("SELECT AVG(totalwords) FROM 'jan'").fetchone()[0]
#reply_avg-7.74
orig_avgj = conn.execute("SELECT AVG(totalwords_o) FROM 'jan-o'").fetchone()[0]
#orig_avg--13.68





###### ACTION WORDS ADDITION ######

# Deliver: 
# Variables added to dataset for each parent tweet and each comment for whether they contain mobilizing information (1) or not (0)
# Report the average number of parent tweets that contained mobilizing calls to action in the entire dataset.
# Report the average number of comments that contained mobilizing calls to action in the entire dataset.
# Report the average number and percentage of comments that contained mobilizing calls to action, but their parent tweets did not (e.g. parent tweet == 0) for the entire dataset. Example: Across the entire dataset, on parent tweets by news outlets that did not contain any mobilizing calls to action, X% of the comments on that tweet contained a call to action. 
# Report the average number of parent tweets that contained mobilizing calls to action in the Jan 6th case study.
# Report the average number of comments that contained mobilizing calls to action in the Jan 6th case study.
# Report the average number and percentage of comments that contained mobilizing calls to action, but their parent tweets did not (e.g. parent tweet == 0) for the Jan 6th case study. Example: On January 6th, on parent tweets by news outlets that did not contain any mobilizing calls to action, X% of the comments on that tweet contained a call to action. 


###one time addition of column
################################################################
###add content from original tweets wordcount to 

#for browsing

#orig_all = pd.read_sql("SELECT * FROM 'tweets-o' WHERE ((' ' || text_o || ' ') LIKE '% think %' OR (' ' || text_o || ' ') LIKE '% must %' OR (' ' || text_o || ' ') LIKE '% need to %' OR (' ' || text_o || ' ') LIKE '% needs to %' OR (' ' || text_o || ' ') LIKE '%$ research %' OR (' ' || text_o || ' ') LIKE '%$ go %' OR (' ' || text_o || ' ') LIKE '%$ call %' OR (' ' || text_o || ' ') LIKE '% vot%' OR (' ' || text_o || ' ') LIKE '%$ act%' OR (' ' || text_o || ' ') LIKE '% try %' OR (' ' || text_o || ' ') LIKE '% you should %' OR (' ' || text_o || ' ') LIKE '% we should %' OR (' ' || text_o || ' ') LIKE '%$ look at %' OR (' ' || text_o || ' ') LIKE '%$ stop %' OR (' ' || text_o || ' ') LIKE '% they should %')", con = cnx)
comp_o = pd.read_sql("SELECT * FROM 'tweets-o'", con = cnx)
exclude = orig_all.conversation_id_o



#origj = pd.read_sql("SELECT * FROM 'jan-o' WHERE ((' ' || text_o || ' ') LIKE '% think %' OR (' ' || text_o || ' ') LIKE '% must %' OR (' ' || text_o || ' ') LIKE '% need to %' OR (' ' || text_o || ' ') LIKE '% needs to %' OR (' ' || text_o || ' ') LIKE '%$ research %' OR (' ' || text_o || ' ') LIKE '%$ go %' OR (' ' || text_o || ' ') LIKE '%$ call %' OR (' ' || text_o || ' ') LIKE '% vot%' OR (' ' || text_o || ' ') LIKE '%$ act%' OR (' ' || text_o || ' ') LIKE '% try %' OR (' ' || text_o || ' ') LIKE '% you should %' OR (' ' || text_o || ' ') LIKE '% we should %' OR (' ' || text_o || ' ') LIKE '%$ look at %' OR (' ' || text_o || ' ') LIKE '%$ stop %' OR (' ' || text_o || ' ') LIKE '% they should %')", con = conn)
comp_oj = pd.read_sql("SELECT * FROM 'jan-o'", con = conn)
excludej = orig_all.conversation_id_o


reply_all = pd.read_sql("SELECT * FROM 'tweets' WHERE ((' ' || text || ' ') LIKE '% think %' OR (' ' || text || ' ') LIKE '% must %' OR (' ' || text || ' ') LIKE '% need to %' OR (' ' || text || ' ') LIKE '% needs to %' OR (' ' || text || ' ') LIKE '%$ research %' OR (' ' || text || ' ') LIKE '%$ go %' OR (' ' || text || ' ') LIKE '%$ call %' OR (' ' || text || ' ') LIKE '% vot%' OR (' ' || text || ' ') LIKE '%$ act%' OR (' ' || text || ' ') LIKE '% try %' OR (' ' || text || ' ') LIKE '% you should %' OR (' ' || text || ' ') LIKE '% we should %' OR (' ' || text || ' ') LIKE '%$ look at %' OR (' ' || text || ' ') LIKE '%$ stop %' OR (' ' || text || ' ') LIKE '% they should %')", con = cnx)
comp = pd.read_sql("SELECT * FROM 'tweets'", con = cnx)
exclude =reply_all[~reply_all.conversation_id.isin(exclude)] 


#replyj = pd.read_sql("SELECT * FROM 'jan' WHERE ((' ' || text || ' ') LIKE '% think %' OR (' ' || text || ' ') LIKE '% must %' OR (' ' || text || ' ') LIKE '% need to %' OR (' ' || text || ' ') LIKE '% needs to %' OR (' ' || text || ' ') LIKE '%$ research %' OR (' ' || text || ' ') LIKE '%$ go %' OR (' ' || text || ' ') LIKE '%$ call %' OR (' ' || text || ' ') LIKE '% vot%' OR (' ' || text || ' ') LIKE '%$ act%' OR (' ' || text || ' ') LIKE '% try %' OR (' ' || text || ' ') LIKE '% you should %' OR (' ' || text || ' ') LIKE '% we should %' OR (' ' || text || ' ') LIKE '%$ look at %' OR (' ' || text || ' ') LIKE '%$ stop %' OR (' ' || text || ' ') LIKE '% they should %')", con = cnx)
#compj = pd.read_sql("SELECT * FROM 'jan'", con = conn)

##len(orig_all)/len(comp_o) #.05

pd.DataFrame(len(reply_all)/len(comp), columns=['mobilize_replies']).to_csv('C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data_22/reply_mobilize_all')


#len(origj)/len(comp_oj) #12%
#len(replyj)/len(compj) #6%

#replyj[~replyj.conversation_id.isin(excludej)] #29796

#22356/len(comp) 




checkpoint = 'C:/Users/dl0ck/OneDrive/Fall 2021/TwitterCarlson/PartyPredictions-main/models/fine_tuned_distilbert'  # Path to model goes here.
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')  # Tokenizer can be adjusted if needed.
pipe = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, truncation=True)
df = pd.read_sql(sql =" SELECT * FROM 'tweets-o'", con = cnx)
df['question'] =df.text_o.str.encode('latin1', 'ignore') 


predictions_o = []
scores_o = []

# -Variable added to dataset for partisan bias for each comment and partisan bias for each parent tweet.
# -Estimate the difference in bias estimate between parent tweet and each comment on that tweet 
 	#(as with word count above). Add this variable to the dataset.
# - Report the average partisan bias for all comments in the dataset. - Report the average partisan bias for all parent tweets in the dataset. 
# - Report the average difference in partisan bias between parent tweets and comments in the dataset.
# - Plot: Nice-looking plot showing the distribution of partisan bias in parent tweets (one line, solid), distribution of partisan bias in comments (one line, dashed), all in black and white and clearly labeled (e.g. score will range from -2 (extremely liberal) to +2 (extremely conservative) – label axes as such). Left panel should show distributions for the full dataset, right panel should show distributions for just January 6th data.
# Mobilizing: Are comments more likely to contain cal
progress = 0
start = time.process_time()
import datetime
now = datetime.datetime.now()
now
# Predict each question and add predictions to column lists.
for q in df['question']:  # Replace 'question' with the proper column name.
    progress +=1
    q = str(q)  # As a precaution.
    pred_dict = pipe(q)[0]  # Gives dictionary of label and score.
    label = pred_dict['label']

    # This part is optional to make the labels more readable.
    # Replace names/shorthands as necessary.
    if label == 'LABEL_0':
        predictions_o.append('D')
    elif label == 'LABEL_1':
        predictions_o.append('R')
    else:
        predictions_o.append('N')  # In case of an invalid label.
    # Entire preceding part can be replaced by predictions.append(label)
    if progress in [1000,2000,3000,4000]:
        print('10000 more down')
    scores_o.append(pred_dict['score'])

elapsed = (time.process_time() - start)
elapsed/60
pd.DataFrame(predictions_o, columns=['bias_pred']).to_csv('C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data_22/orig_kauf_preds_all')
pd.DataFrame(scores_o, columns=['bias_score']).to_csv('C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data_22/orig_kauf_score_all')
# pd.DataFrame(predictions_o, columns=['bias_pred']).to_csv('C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data_22/orig_kauf_preds_jan')
# pd.DataFrame(scores_o, columns=['bias_score']).to_csv('C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data_22/orig_kauf_score_jan')

jan_o = pd.read_sql(sql =" SELECT * FROM 'tweets-o'", con = cnx)
tweets['score_o'] = pd.read_csv('C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data_22/orig_kauf_score_all').bias_score
tweets['pred_o'] = pd.read_csv('C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data_22/orig_kauf_preds_all').bias_pred

tweets.score_o.max()
tweets.score_o.min()
tweets.score_o.mean()

grouped_orig = tweets.groupby("conversation_id_o")
mean_orig = grouped_orig.mean().score.mean()




df = pd.read_sql(sql =" SELECT * FROM 'jan'", con = conn)
df['question'] =df.text.str.encode('latin1', 'replace') 
predictions = []
scores = []
progress = 0
start = time.process_time()

for q in df['question']:  # Replace 'question' with the proper column name.
    progress+=1
    q = str(q)  # As a precaution.
    pred_dict = pipe(q)[0]  # Gives dictionary of label and score.
    label = pred_dict['label']

    # This part is optional to make the labels more readable.
    # Replace names/shorthands as necessary.
    if label == 'LABEL_0':
        predictions.append('D')
    elif label == 'LABEL_1':
        predictions.append('R')
    else:
        predictions.append('N')  # In case of an invalid label.
    # Entire preceding part can be replaced by predictions.append(label)
    if progress in [10000,20000,30000,40000]:
        print('10000 more down')


    scores.append(pred_dict['score'])
conn.close()
elapsed = (time.process_time() - start)
elapsed/60


pd.DataFrame(predictions, columns=['bias_score']).to_csv('C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data_22/reply_kauf_preds_jan')
pd.DataFrame(scores, columns=['bias_score']).to_csv('C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data_22/reply_kauf_score_jan')

conn = create_engine('sqlite:///C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data_22/processed/jan_clean.db').connect()


jan = pd.read_sql(sql =" SELECT * FROM 'jan'", con = conn)
tweets = pd.read_sql(sql =" SELECT * FROM 'tweets'", con = cnx)

jan['score'] = pd.read_csv('C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data_22/reply_kauf_score_jan').bias_score
tweets['score'] = pd.read_csv('C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data_22/reply_kauf_score_all').bias_score

jan['preds'] = pd.read_csv('C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data_22/reply_kauf_preds_jan').bias_score
all['preds'] = pd.read_csv('C:/Users/Dl0ck/OneDrive/Fall 2021/TwitterCarlson/data_22/reply_kauf_preds_all').bias_score


jan.score.max()
all.score.max()

jan.score.min()
all.score.min()

jan.score.mean()
all.score.mean()


grouped_reply = jan.groupby("conversation_id")
grouped_replya = all.groupby("conversation_id")

mean_reply = grouped_reply.mean().score.mean()
mean_replya = grouped_replya.mean().score.mean()


fig,ax = plt.subplots(figsize=(12,4))# Add labels and title

sb.set(style = 'whitegrid')
sb.distplot(jan_o.score_o, hist=False, kde_kws={'linestyle':'--'}, label ='Parent Tweet',color ='black')
sb.distplot(jan.score, hist=False, label ="Reply",color = 'grey')
plt.title("Distribution of partisan bias in tweets and replies")
# ax.set_xticks(range(-2,3))
# ax.set_xticklabels(range(-2,3))
# ax.set_xticklabels(['-2 \nExtremely \nLiberal','-1','0 \nNeutral','1','2 \nExtremely \nConservative'])
# ax.set_xticklabels(['0.3 ','0.4\nMore \nLiberal','0.5','0.6','0.7','0.8','0.9','1','1.1 \nMore\nConservative' ])

ax.set_xlabel('Partisan bias prediction') 
plt.legend()
plt.show()