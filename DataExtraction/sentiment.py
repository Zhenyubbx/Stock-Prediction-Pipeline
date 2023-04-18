import tweepy
from textblob import TextBlob
from langdetect import detect
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import nltk


def sentiment_score_transform():
    data = pd.read_csv("cleaned_hashtag_tesla_TweetTexts.csv")
    data['Tweet Text'] = data['Tweet Text'].astype(str) #Change the tweet data type from object to string
    data.dropna(subset=['Tweet Text'], inplace=True) #drop rows with na values in Tweet column
    data.reset_index(drop=True, inplace=True) #reset indexes and drop the old index column
    sia = SentimentIntensityAnalyzer()
    data['Sentiments'] = data['Tweet Text'].apply(lambda Tweet: sia.polarity_scores(Tweet))
    data = pd.concat([data.drop(['Sentiments'], axis=1), data['Sentiments'].apply(pd.Series)], axis=1)
    data.to_gbq("is3107-project-383009.Dataset.tslaStockAnalysed", project_id="is3107-project-383009")
    print(data)
    return data


sentiment_score_transform()

