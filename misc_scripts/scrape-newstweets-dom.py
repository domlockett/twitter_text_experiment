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
import decimal
import os
import json
import pandas
import csv
from datetime import datetime, timedelta
import datetime as dt
import numpy as np
from time import sleep
import logging
import sys
import ast
from sqlalchemy import create_engine
from sqlalchemy.types import String
import sqlalchemy
# Functions for request logistics ------------
# Function to grab token from environment
def auth():
    return os.environ.get("BEARER_TOKEN")

# Function to create header

def connect_to_endpoint_noparams(url, headers):
    response = requests.request("GET", url, headers=headers)
    #print(response.status_code)
    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )
    return response.json()

def connect_to_endpoint_fullarchive(headers, params):
    search_url = "https://api.twitter.com/2/tweets/search/all"
    response = requests.request("GET", search_url, headers=headers, params=params)
    #print(response.status_code)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


# Functions that format specific requests ------------

# NOTE:
# Tweet fields are adjustable.
# Options include:
# attachments, author_id, context_annotations,
# conversation_id, created_at, entities, geo, id,
# in_reply_to_user_id, lang, non_public_metrics, organic_metrics,
# possibly_sensitive, promoted_metrics, public_metrics, referenced_tweets,
# source, text, and withheld

# User fields are adjustable.
# Options include:
# created_at, description, entities, id, location, name,
# pinned_tweet_id, profile_image_url, protected,
# public_metrics, url, username, verified, and withheld

# URL to grab news sources' user-level information
# (Need this function mostly for user_id)
def create_url_user_lookup(usernames):
    user_fields = "user.fields=id,public_metrics"
    url = "https://api.twitter.com/2/users/by?{}&{}".format(usernames, user_fields)
    return url

# Specify what tweet-level information I want from news source
def get_params_user_tweets(user_id, pagination_token, start_time, end_time):
    user = 'from:%s' % user_id
    return {'query': user, 'tweet.fields': "created_at,text,conversation_id,in_reply_to_user_id,public_metrics,referenced_tweets,lang,author_id,context_annotations,attachments",
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
    df = pandas.json_normalize(json_obj).astype('str')
    if(tweet_id != "None"): df["tweet_id"] = tweet_id #for special case of writing context annotations
    missing_cols = list(set(colnames).difference(df.columns.tolist())) #any mising columns?
    for c in missing_cols: df[c] = None #if so, fill in as empty
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

# Function to get news sources' ids
def get_user_info(headers, filename):
    logging.info('Getting user-level info')
    # Specify the usernames to lookup info for
    # You can enter up to 100 comma-separated values.
    # See Notes/NewsOutletList.pdf for removed outlets
    usernames = "usernames=ABC"
    url_user_lookup = create_url_user_lookup(usernames)
    try:
        json_response = connect_to_endpoint_noparams(url_user_lookup, headers)
    except Exception as e:
        logging.exception("get_user_info() error: %s", e)
    # Write out user-level data
    user_df = pandas.json_normalize(json_response["data"])
    user_df.to_csv('data/users/users ' + filename, index = False)
    return user_df["id"]

# Function to get each news source's timeline of tweets,
# including media and context annotations
# Paginate through results and write to csv
def get_user_tweets(headers, user_ids, start_time, end_time):
    logging.info('Getting users Tweet-level info')
    # create structure of Tweet-level csv
    tweet_colnames = ['id','conversation_id','author_id', 'lang','created_at', 'text',
        'in_reply_to_user_id','referenced_tweets', 'public_metrics.retweet_count',
        'public_metrics.reply_count', 'public_metrics.like_count',
        'public_metrics.quote_count', 'attachments.media_keys']
    # create structure of Media-level csv
    media_colnames = ['media_key', 'type', 'duration_ms', 'preview_image_url', 'public_metrics.view_count']
    # create structure of context csv
    context_colnames = ['tweet_id', 'domain.id', 'domain.name', 'entity.id', 'entity.name']

    for i in user_ids:
        logging.info(i)
        #url_user_tweets = create_url_user_tweets(i)
        pagination_token = None
        stop = False
        while stop == False:
            # get max number of Tweets
            params_user_tweets = get_params_user_tweets(user_id = i,
                pagination_token = pagination_token,
                start_time = start_time,
                end_time = end_time)
            try:
                json_response = connect_to_endpoint_fullarchive(headers, params_user_tweets)
                sleep(3) # 300 requests to this endpoint / 15 minutes
            except Exception as e:
                logging.exception("get_user_tweets() error: %s", e)
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


def main():
    logging.basicConfig(filename='data/logs/' + datetime.today().strftime("%m-%d-%Y") + '.log',
        level=logging.INFO, format='%(message)s')

    logging.info('Start time: %s', datetime.now())

    start_time = "2017-01-01T00:00:00Z"
    end_time = "2017-01-02T00:00:00Z"

    logging.info('Testing: %s through %s', start_time, end_time)

    # Setup API authentication
    bearer_token = auth()
    headers = create_headers(bearer_token)
    filename = create_filename()


    # 1. Get news source's twitter ID
    user_ids = get_user_info(headers,filename)

    # 2. Get each news source's timeline of Tweets for a given time span
    get_user_tweets( headers, user_ids,  start_time,  end_time)

    logging.info('End time: %s', datetime.now())


main()


# Note:
#
# Requests per 15 minutes
# - User lookup -- 300
# - User Tweet timeline -- 1500
# - Recent search -- 450
# - Full-archive search -- 300 (1 request / 1 second)
#           300 requests / 15 minutes -- 500 results per request 
#
# But, Recent search, user Tweet timeline, user mention timeline,
# and filtered stream share a Project-level Tweet cap limit of 500,000 Tweets per month. 
#
# (Documentation: https://developer.twitter.com/en/docs/twitter-api/rate-limits)




