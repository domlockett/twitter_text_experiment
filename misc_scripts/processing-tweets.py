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

stop = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've"]





# A similar approach with sentiment. I want to get at whether the comments on a tweet are more negative than the original tweet. 
# So, I think we'd want the average sentiment of the comment thread for a give tweet compared to the sentiment of that tweet. I'm open to different measures of sentiment. Some off-the-shelf measures are probably fine to start with, but eventually we'll probably want to get a bit fancier.
# Similarity. I think we could try to estimate the similarity (can start with simple cosine sim, unless you have a preferred measure) between each comment and its parent tweet. 
# So, if there are 100 comments on a tweet, what's the similarity score between comment 1 and the tweet? What's the score between comment 2 and the tweet? Comment 3 and the tweet? and so on. Then, look at the average of these for each tweet.

#to do:
#Number of words |X|
#Average numbre of words per comment |X|
#Sentiment of the tweet|X|
#Date/time stamp |X|
#Link to news article |X|
#Text of the news article|X|
#News outlet for the original tweet |X|

# Conditional import functions
def importIDS():
	# load in unique ids
	global tweetIDS
	global tweetconIDS
	global tweetmedIDS
	cnn = create_engine('sqlite:///data/processed/tweetid.db').connect()		
	tweetIDS = pd.read_sql('tweetIDS', cnn)
	tweetconIDS = pd.read_sql('tweetconIDS', cnn)
	tweetmedIDS = pd.read_sql('tweetmedIDS', cnn)
	cnn.close()


#importIDS()


#49238
allSets = []
def importCSV():
	try:
		files = [os.listdir(os.getcwd() + '\\data\\tweets-context'), os.listdir(os.getcwd() + '\\data\\tweets-media')]#grab all the files from two other tweet folders
		fileTweets = os.listdir(os.getcwd() + '\\data\\tweets')#set up original tweet folder as index
		ref = ['-' + re.sub(r'[a-zA-z\s\.+$()@]','' , i) for i in fileTweets]#pull date only from tweet
		for index, item in enumerate(ref[0:5]):
			fileSet = [fileTweets[index]]#init list of three [tweets, tweetsmedia,tweetscontext]
			for number, content in enumerate(files):
				try:
					fileDates = [re.sub(r'[a-zA-z\s\.+$()@]', '', i) for i in files[number]]#grab date only from other 2 folders
					fileName = files[number][fileDates.index(list(filter(lambda a: ref[index] in a, fileDates))[0])]#look for matches to single reference (tweets.csv) file in each folder and save file name
				except:
					fileSet.append(np.nan) # if at any point a match doesnt exist fill with NA in list of 3
				else:
					fileSet.append(fileName) #otherwise fill with the file name
			allSets.append(fileSet) # add list of 3 to list
	except Exception as e:
		logging.exception("importData() error: %s", e)
	return allSets

def importData(fileSet):
	try:
		global tweets
		global tweetsContext
		global tweetsMedia
		tweets = pd.read_csv('data\\tweets\\' + fileSet[0])
		tweetsContext = pd.read_csv('data\\tweets-context\\' + fileSet[1])
		tweetsMedia = pd.read_csv('data\\tweets-media\\' + fileSet[2])
	except Exception as e:
		logging.exception("importData() error: %s", e)




def importSQl():
	try:
		global tweets
		global tweetsContext
		global tweetsMedia
		engine = create_engine('sqlite:///data/sql/tweet.db')
		cnx = engine.connect()
		metadata = sq.MetaData()
		t = sq.Table('tweets', metadata, autoload_with=engine)
		query = sq.select([t]).where(t.columns.id.notin_(tweetIDS.tweet_id.tolist()))
		result = cnx.execute(query).all()
		tweets = pd.DataFrame(result)
		tweets.columns = result[0].keys()

		tm = sq.Table('tweets-media', metadata, autoload_with=engine)
		query2 = sq.select([tm]).where(tm.columns.media_key.notin_(tweetmedIDS.unique_id.tolist()))
		result2 = cnx.execute(query2).all()
		tweetsMedia = pd.DataFrame(result2)
		tweetsMedia.columns = result2[0].keys()

		
		tc = sq.Table('tweets-context', metadata, autoload_with=engine)
		query3 = sq.select([tc]).where(tc.columns.tweet_id.notin_(tweetconIDS.unique_id.tolist()))
		result3 = cnx.execute(query3).all()
		tweetsContext = pd.DataFrame(result3)
		tweetsContext.columns = result3[0].keys()
  
		tweets = tweets[:1300]
		tweetsMedia = tweetsMedia[:1300]
		tweetsContext = tweetsContext[:1300]
		cnx.close()
  
	except Exception as e:
			logging.exception("importSQL() error: %s", e)

#gloveFile = "glove.6B.50d.txt"

def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    with open(gloveFile, encoding="utf8" ) as f:
        content = f.readlines()
    model = {}
    for line in content:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model
   
# Processing functions
def transform():
	try:
		#remove periods because they are functional in python
		tweets.columns = [re.sub(r'\.', "",c) for c in tweets.columns]
		tweets.rename(columns={'attachmentsmedia_keys': 'media_key', 'id':'tweet_id', 'public_metricsretweet_count': 'retweet_count','public_metricsreply_count': 'reply_count','public_metricslike_count': 'like_count','public_metricsquote_count':'quote_count'}, inplace=True)#change media keys to match xxx-media
		tweets['media_key'] = tweets['media_key'].str.strip('[]').str.strip('\'')#remove brackets and quotes
		tweets['tweet_id'] = tweets['tweet_id'].astype(str)#change id to string, int too intensive


		tweetsContext.columns = [re.sub(r'\.', "",c) for c in tweetsContext.columns]
		tweetsContext.rename(columns={'domainid':'domain_id','domainname':'domain_name','entityid':'entity_id','entityname':'entity_name'})
		tweetsContext['unique_id'] = [datetime.now().strftime('%Y%m%d%H%M%S%f')+ str(int(uuid.uuid4())) for _ in range(len(tweetsContext))]
		tweetsContext['tweet_id'] = tweetsContext['tweet_id'].astype(str) #force other ids to str

		tweetsMedia.columns = [re.sub(r'\.', "",c) for c in tweetsMedia.columns]
		tweetsMedia.rename(columns = {'public_metricsview_count':'media_view_count'})
		tweetsMedia['unique_id'] = [datetime.now().strftime('%Y%m%d%H%M%S%f')+ str(int(uuid.uuid4())) for _ in range(len(tweetsMedia))]


	except Exception as e:
		logging.exception("transform() error: %s", e)

#outgroup measures of negative partisanship?
#analyse twitter use in congress

def sentiment():
	try:
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
		tweets['subjectivity'] = tweets['text'].apply(getSubjectivity)
		tweets['polarity'] = tweets['text'].apply(getPolarity)
		tweets['analysis'] = tweets['polarity'].apply(getAnalysis)
	except Exception as e:
		logging.exception("sentiment() error: %s", e)

def getAnalysis(score):
			if score < 0:
				return 'Negative'
			elif score == 0:
				return 'Neutral'
			else:
				return 'Positive'

def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

def process():
	try:
		
		tweets['avg_words'] = tweets['text'].apply(lambda x: avg_word(x))
		tweets['text']=tweets['text'].fillna('').apply(str)
		tweets['links'] = tweets['text'].str.extract(r"(http\S+)")#extract any links from text into new column
		tweets['text'] = tweets['text'].str.replace(r"(http\S+)","").replace(np.nan, "")
		tweets['totalwords'] = [len(x.split()) for x in tweets['text'].tolist()]#word count to new column
		tweets.text = [" ".join((str.lower(word)) for word in x.split()) for x in tweets.text]#Change to lowercase
		tweets['hashtags'] = tweets['text'].apply(lambda x: re.findall(r'\B#\w*[a-zA-Z]+\w*', x)) # new column #s
		tweets['mentions'] = tweets['text'].apply(lambda x: re.findall(r'\B@\w*[a-zA-Z]+\w*', x)) #new column @s
		tweets['avg_count_convo'] = tweets.groupby('conversation_id')['totalwords'].transform('mean')
		sentiment()
		tweets['avg_polar_convo'] = tweets.groupby('conversation_id')['polarity'].transform('mean')
		tweets['text'] = tweets['text'].astype('str')
		tweets['avg_sent_convo'] = tweets['polarity'].apply(getAnalysis)


  
  
		tweets.text = [" ".join(re.sub(r"#(\w+)", '',word) for word in x.split()) for x in tweets['text'].tolist()]#remove #s
		tweets.text = [" ".join(re.sub(r"@(\w+)", '',word) for word in x.split()) for x in tweets['text'].tolist()]#remove @s
		tweets.text = tweets['text'].str.replace('[^\w\s]','') #drop punctuation
		tweets.text = [" ".join(x for x in x.split() if x not in stop) for x in tweets.text.tolist()] # remove sstopwords'
		tweets.text = [" ".join(str(TextBlob(word).correct()) for word in x.split()) for x in tweets.text.tolist()]
	except Exception as e:
		logging.exception("process() error: %s", e)
	return


# def cosine_method1(s1):
#     vector_1 = np.mean([model[word] for word in preprocess(s1)],axis=0)
# #     vector_2 = np.mean([model[word] for word in preprocess(s2)],axis=0)
# #     cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
# tweets['text'].apply(lambda x: cosine_method1(x))
    
#     vector1 = np.mean([model[word] for word in tweets['vector_1'].tolist()],axis=0)
#     #vector_2 = np.mean([model[word] for word in preprocess(s2)],axis=0)
#     #cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
#     #print('Word Embedding method with a cosine distance asses that our two sentences are similar to',round((1-cosine)*100,2),'%')
    
# def cosine_method2():
#     letters_only_text = re.sub("[^a-zA-Z]", " ", tweets.text)
#     letters_only_text = re.sub(r'\n',r' ', letters_only_text)
#     #vector_1 = np.mean([model[word] for word in preprocess(s1)],axis=0)
#     vector1 = np.mean([model[word] for word in letters_only_text],axis=0)
#     #cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
#     #print('Word Embedding method with a cosine distance asses that our two sentences are similar to',round((1-cosine)*100,2),'%')



#Exporting functions
def completeIDTWEET():
	try:
		t = tweets['tweet_id']
		tc = tweetsContext['unique_id']
		tm = tweetsMedia['unique_id']
		cnx = create_engine('sqlite:///data/processed/tweetidTWEET.db', echo = False).connect()
		t.to_sql("tweetIDS",cnx, if_exists='append', index=False)
		tc.to_sql("tweetconIDS", cnx, if_exists='append', index=False)
		tm.to_sql("tweetmedIDS", cnx, if_exists='append', index=False)
	except Exception as e:
		logging.exception("completeID() error: %s", e)

def exportSQLTWEET():
	try:
		cnx = create_engine('sqlite:///data/processed/cleanTWEET.db').connect()
		tweets['analysis'] = tweets['analysis'].astype('str')
		tweets['hashtags'] = tweets['hashtags'].astype('str')
		tweets['mentions'] = tweets['mentions'].astype('str')
		tweets['tweet_id'] = tweets['tweet_id'].astype('str')
		tweetsMedia['unique_id'] = tweetsMedia['unique_id'].astype('str')
		tweetsContext['unique_id'] = tweetsContext['unique_id'].astype('str')



		tweets.to_sql("tweets", cnx, if_exists='append', index=False)
		tweetsContext.to_sql("tweetsContext", cnx, if_exists='append', index=False)
		tweetsMedia.to_sql("tweetsMedia", cnx, if_exists='append', index=False)
		cnx.close()
	except Exception as e:
		logging.exception("exportSQL() error: %s", e)
	return



#Exporting functions
def completeIDCOMM():
	try:
		t = tweets['tweet_id']
		tc = tweetsContext['unique_id']
		tm = tweetsMedia['unique_id']
		cnx = create_engine('sqlite:///data/processed/tweetidCOMM.db', echo = False).connect()
		t.to_sql("tweetIDS",cnx, if_exists='append', index=False)
		tc.to_sql("tweetconIDS", cnx, if_exists='append', index=False)
		tm.to_sql("tweetmedIDS", cnx, if_exists='append', index=False)
	except Exception as e:
		logging.exception("completeID() error: %s", e)

def exportSQLCOMM():
	try:
		cnx = create_engine('sqlite:///data/processed/cleanCOMM.db').connect()
		tweets['analysis'] = tweets['analysis'].astype('str')
		tweets['hashtags'] = tweets['hashtags'].astype('str')
		tweets['mentions'] = tweets['mentions'].astype('str')
		tweets['tweet_id'] = tweets['tweet_id'].astype('str')
		tweetsMedia['unique_id'] = tweetsMedia['unique_id'].astype('str')
		tweetsContext['unique_id'] = tweetsContext['unique_id'].astype('str')



		tweets.to_sql("tweets", cnx, if_exists='append', index=False)
		tweetsContext.to_sql("tweetsContext", cnx, if_exists='append', index=False)
		tweetsMedia.to_sql("tweetsMedia", cnx, if_exists='append', index=False)
		cnx.close()
	except Exception as e:
		logging.exception("exportSQL() error: %s", e)
	return


def runall():
    transform()
    process()
    sentiment()

    
def main(importType):#run processing functions here
	if importType == 'csv':
		allSets = importCSV()
		for i in allSets:
			importData(i)
			runall()
			completeIDTWEET()
			exportSQLTWEET()
   

	elif importType == 'sql':
		importIDS()
		importSQl()
		runall()
		completeIDCOMM()
		exportSQLCOMM()


import time
start_time = time.time()
main('sql')

print("--- %s seconds ---" % (time.time() - start_time))

#^ basically for future tweet collection     
#main('csv') 




