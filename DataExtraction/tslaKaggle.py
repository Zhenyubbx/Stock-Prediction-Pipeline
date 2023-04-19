
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
from datetime import datetime

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv("tesla_kaggle.csv")

# Keep only the "Tweet Text" and "Date & Time" columns
df = df[["Date & Time", "Tweet Text"]]

# Define a function to filter out invalid dates
def filter_valid_dates(date_str):
    try:
        datetime.strptime(date_str, '%B %d, %Y at %I:%M%p')
        return True
    except ValueError:
        return False

# Apply the filter function to the 'date' column
valid_dates_mask = df['Date & Time'].apply(filter_valid_dates)
df = df[valid_dates_mask]


# Convert date string to datetime object
df["Date & Time"] = pd.to_datetime(df["Date & Time"], format="%B %d, %Y at %I:%M%p")


# Format Date & Time in "yyyy-mm-dd" format
df["Date & Time"] = df["Date & Time"].dt.strftime("%Y-%m-%d")


# Remove URLs, mentions, and hashtags from the Tweet Text text
df["Tweet Text"] = df["Tweet Text"].str.replace(r"http\S+", "")
df["Tweet Text"] = df["Tweet Text"].str.replace(r"@\S+", "")
df["Tweet Text"] = df["Tweet Text"].str.replace(r"#\S+", "")

# Remove leading and trailing whitespace from the Tweet Text text
df["Tweet Text"] = df["Tweet Text"].str.strip()
df.reset_index(inplace=False)
df = df.rename(columns={"Tweet Text": "Tweet", "Date & Time" : "Date"})

df = df.dropna()
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(by='Date', ascending=False)

# Write the cleaned data to a new CSV file
df.to_csv("cleaned_hashtag_tesla_TweetTexts.csv", index=False)




def sentiment_score_transform():
    data = pd.read_csv("cleaned_hashtag_tesla_TweetTexts.csv")
    data['Tweet Text'] = data['Tweet Text'].astype(str) #Change the tweet data type from object to string
    data['Date & Time'] = data['Date & Time'].astype(str) #Change the tweet data type from object to string
    data.dropna(subset=['Tweet Text', 'Date & Time'], inplace=False) #drop rows with na values in Tweet and Date columns
    data["Tweet Text"] = data["Tweet Text"].str.strip()
    data.reset_index(inplace=False)
    sia = SentimentIntensityAnalyzer()
    data['Sentiments'] = data['Tweet Text'].apply(lambda Tweet: sia.polarity_scores(Tweet))
    data = pd.concat([data.drop(['Sentiments'], axis=1), data['Sentiments'].apply(pd.Series)], axis=1)
    data.to_gbq("is3107-project-383009.Dataset.tslaStockAnalysed1", project_id="is3107-project-383009")
    


sentiment_score_transform()



df.to_gbq("is3107-project-383009.Dataset.tslaTweetsKaggle", project_id="is3107-project-383009")
