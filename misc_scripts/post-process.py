import pandas as pd
import os
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
#import glove
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

#Difference between the two-
    #Soroka & Young dictionary
    #QUantita 
    #supervised measures of sentiment
#Dynamic topic model (Erin)-
    #flag the tweets with what topic the tweets are about
    #Structural topic model -- maybe too hard
    #Some 'tasks are more global'
    #Topic switching is for later


file = open(os.getcwd() + '\\data\\url.txt',encoding = 'utf-8')
url = file.read()
file.close()

# DIFFERENCE IN SENTIMENT
# TRAINING + COSINE SIMILIARITY
model = loadGloveModel(gloveFile)
heat_map_matrix_between_two_sentences(whole['text'][1], whole['text_comm'][1])
def cosine_distance_wordembedding_method(s1, s2):
    vector_1 = np.mean([model[word] for word in preprocess(s1)],axis=0)
    vector_2 = np.mean([model[word] for word in preprocess(s2)],axis=0)
    cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
    print('Word Embedding method with a cosine distance asses that our two sentences are similar to',round((1-cosine)*100,2),'%')

#cosine_distance_wordembedding_method(whole['text'][1], whole['text_comm'][1])