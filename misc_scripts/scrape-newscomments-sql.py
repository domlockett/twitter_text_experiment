# Run in terminal before running Python
# (Mac) 
# export 'BEARER_TOKEN'='your own token here'
# 
# (Windows)
# SET BEARER_TOKEN=your token here
#
# Navigate to 'Box/TwitterConversations' folder.  It also has 'data' folder
# and a folder for each of the following:
# - users
# - tweets
# - tweets_media
# - tweets_context
# - comments
# - comments_media

# Using Python 3.7.7
import requests
import os
import json
import pandas
import csv
from datetime import datetime, timedelta
import numpy as np
from time import sleep, time
import logging
import sys

from sqlalchemy import create_engine
from sqlalchemy.types import String
import sqlalchemy


# To download file contents from online Box
# https://github.com/box/box-python-sdk
# pip install "boxsdk[jwt]"

# TODO: Finish this.
# from boxsdk import JWTAuth

# auth = JWTAuth(
#     client_id='m47uczrkqg0o5kw8zati8hocoas7z1o5',
#     client_secret='q1NCE2un43xRRlpZYFRFhSiF28uQwOvc',
#     enterprise_id='280321',
#     jwt_key_id='gakrsj7h',
#     rsa_private_key_file_sys_path='CERT.PEM',
#     rsa_private_key_passphrase='10e7369765f8cc53e92093db10d598d3',
#     store_tokens=your_store_tokens_callback_method,
# )

# access_token = auth.authenticate_instance()



# Functions for request logistics ------------

# Function to grab token from environment
def auth():
    return os.environ.get("BEARER_TOKEN")

# Function to create header
def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers

#https://developer.twitter.com/en/docs/twitter-api/tweets/search/api-reference/get-tweets-search-all
def connect_to_endpoint_fullarchive(headers, params):
    search_url = "https://api.twitter.com/2/tweets/search/all"
    response = requests.request("GET", search_url, headers=headers, params=params)
    #print(response.status_code)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()




# Functions that format specific requests ------------

def get_params_comments(convo_id, pagination_token, start_time, end_time):
    c = 'conversation_id:%s' % convo_id
    return {'query': c, 'tweet.fields': "created_at,text,conversation_id,in_reply_to_user_id,public_metrics,referenced_tweets,lang,author_id,context_annotations,attachments",
        "expansions" : "attachments.media_keys",
        "media.fields" : "duration_ms,preview_image_url,public_metrics", 
        "max_results": 500,
        "next_token": pagination_token,
        "start_time": start_time, "end_time": end_time}

def create_filename():
    # datetime object containing current date and time
    now = datetime.now()
    # string format - mm/dd/YY H.M
    now_string = now.strftime("%m-%d-%Y %H.%M")
    return now_string + ".csv"



# Functions that make requests and save info that is returned --------------------

def write_tweet_data(json_obj, colnames, database, datatype, tweet_id = None):
    df = pandas.json_normalize(json_obj) #.astype('str')
    if(tweet_id != "None"): df["tweet_id"] = tweet_id #for special case of writing context annotations
    missing_cols = list(set(colnames).difference(df.columns.tolist())) #any mising columns?
    for c in missing_cols: df[c] = None #if so, fill in as empty
    if datatype == "tweets":
        df["referenced_tweets"] = df["referenced_tweets"].astype('str')
        df["attachments.media_keys"] = df["attachments.media_keys"].astype('str')
    # Open connections and save to sql databases
    df = df.replace({np.nan: None})
    engine = create_engine('sqlite:///data/sql/' + database)
    sqlite_table = datatype #comments/context/media
    sqlite_connection = engine.connect() #open connection
    df[colnames].to_sql(sqlite_table,sqlite_connection, if_exists='append', index = False)
    sqlite_connection.close()#close connection


def paginate(json_response):
    if "next_token" in json_response["meta"].keys():
        pagination_token = json_response["meta"]["next_token"]
        stop = False
    else:
        pagination_token = None
        stop = True
    return stop, pagination_token


# Function to get each news source's timeline of tweets,
# including media and context annotations
# Paginate through results and write to csv
def get_comments(headers, filename, start_time, end_time, start_idx, end_idx):
    # create structure of Tweet-level csv
    tweet_colnames = ['id','conversation_id','author_id', 'lang','created_at', 'text',
       'in_reply_to_user_id','referenced_tweets', 'public_metrics.retweet_count',
       'public_metrics.reply_count', 'public_metrics.like_count',
       'public_metrics.quote_count', 'attachments.media_keys']

    # create structure of Media-level csv
    media_colnames = ['media_key', 'type', 'duration_ms', 'preview_image_url', 'public_metrics.view_count']

    # create structure of context csv
    context_colnames = ['tweet_id', 'domain.id', 'domain.name', 'entity.id', 'entity.name']

    # text file holding unique conversation_id's that
    # are associated with tweets with >= 1 replies
    # N=3976122
    with open("conversation_ids04-23-21.txt") as f:
        convo_ids = f.readlines()
    convo_ids = [x.strip() for x in convo_ids] 

    for idx, i in enumerate(convo_ids[start_idx:end_idx], start = start_idx):
        pagination_token = None
        stop = False
        while stop == False:
            # get max number of Tweets
            params_user_tweets = get_params_comments(convo_id = i,
                pagination_token = pagination_token,
                start_time = start_time,
                end_time = end_time)
            try:
                #logging.info("Time: %s", datetime.now())
                json_response = connect_to_endpoint_fullarchive(headers, params_user_tweets)
                sleep(3.1) # 300 requests to this endpoint / 15 minutes
            except Exception as e:
                logging.exception("get_comments() error: %s", e)
                if(e.args[0] == 503):
                    sleep(5*60) # sleep for 5 minutes, twitter over capacity
                    logging.info('503 error: %s', datetime.now())
                    continue
                if(e.args[0] == 429):
                    sleep(60) # wait one minute
                    logging.info('429 error: %s', datetime.now())
                    continue 
                else:
                    break

            # break if no more tweets
            if json_response["meta"]["result_count"] == 0:
                break

            # otherwise, write Tweet-level data to file
            write_tweet_data(json_obj = json_response["data"], 
                colnames = tweet_colnames, 
                database = 'tweet.db',
                datatype = 'tweets')

            # then, write Media-level data to file
            if "includes" in json_response.keys():
                write_tweet_data(json_obj = json_response["includes"]["media"], 
                    colnames = media_colnames, 
                    database = 'tweet.db',
                    datatype = 'tweets-media')

            #then, write context data to file
            for j in json_response["data"]:
                if "context_annotations" in j.keys():
                    write_tweet_data(json_obj = j["context_annotations"], 
                        colnames = context_colnames, 
                        database = 'tweet.db',
                        datatype = 'tweets-context',
                        tweet_id = j["id"])

            # paginate 
            stop, pagination_token = paginate(json_response)
        # log when its *done* scraping for that Tweets
        logging.info('idx: %s id: %s', idx, i)


def main(start_idx, end_idx):
    logging.basicConfig(filename='data/logs/' + datetime.today().strftime("%m-%d-%Y") + '.log',
        level=logging.INFO, format='%(message)s')

    logging.info('Start time: %s', datetime.now())

    # Manually set the date and time to wide range.
    # Needed to get tweets beyond last 30 days (default)
    start_time = "2010-01-01T00:00:00Z"
    end_time = "2021-04-01T00:00:00Z"

    # Setup API authentication
    bearer_token = auth()
    headers = create_headers(bearer_token)

    # Create a common file name
    filename = create_filename()

    # Get all comments associated with a range of conversation ids
    get_comments(headers, filename, start_time, end_time, start_idx, end_idx)

    logging.info('End time: %s', datetime.now())

# For example,
# First person 0:100
# Second person 100:200
start_idx = 1000000
end_idx = 1500000
main(start_idx, end_idx)


# # # Check
# cnx = create_engine('sqlite:///data/sql/tweet.db').connect()
# df1 = pandas.read_sql_table('tweets', cnx)
# df2 = pandas.read_sql_table('tweets-media', cnx)
# df3 = pandas.read_sql_table('tweets-context', cnx)
# cnx.close()


