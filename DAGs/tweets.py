import requests
import os
import json
import csv
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import yfinance as yf
import pandas as pd
import re
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from textblob import TextBlob
from langdetect import detect


####################################################
# 1. DEFINE PYTHON FUNCTIONS
####################################################


##EXTRACTION FROM TWITTER
bearer_token = "AAAAAAAAAAAAAAAAAAAAACYhmwEAAAAAl7HpRkqggtVjTCj0LmMI1nMYaQg%3Dp6OKovTzZ4XI6GuDfiscptdLvv35yp9LZNnkTShVYTwblI6ARa"

search_url = "https://api.twitter.com/2/tweets/search/recent"

def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2RecentSearchPython"
    return r

def connect_to_endpoint(url, params):
    response = requests.get(url, auth=bearer_oauth, params=params)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()




    
def process_tweets():
    current_time = datetime.utcnow()
    ten_seconds = timedelta(seconds=10)
    end_time = current_time - ten_seconds

    # Optional params: start_time,end_time,since_id,until_id,max_results,next_token,
    # expansions,tweet.fields,media.fields,poll.fields,place.fields,user.fields
    query_params = {'query': '#tsla', 'max_results': '100', 'tweet.fields': 'created_at,author_id'}

    tweets = []

    
    for page in range(100):
        query_params['next_token'] = None
        json_response = connect_to_endpoint(search_url, query_params)
        tweets += json_response['data']
        print(f"Successfully pulled {len(json_response['data'])} tweets from Tweepy! Page {page+1}")

        if 'next_token' in json_response['meta']:
            query_params['next_token'] = json_response['meta']['next_token']
        else:
            break

    with open('tsla_tweets.json', mode='w') as file:
        data_list = []

        for tweet in tweets:
            date = datetime.strptime(str(tweet['created_at'].replace("Z", "")), '%Y-%m-%dT%H:%M:%S.000')
            data_list.append({
                'Date & Time': str(date.strftime('%Y-%m-%d')),
                'Tweet Text': tweet['text']
            })

        combined = pd.DataFrame(data=data_list)

        def cleanText(text):
            text = re.sub(r'@[A-Za-z0-9]+', '', text) #Removes @mentions
            text = re.sub(r'#', '', text) #Removes the # symbol
            text = re.sub(r'RT[\s]+', '', text) #Removes RT
            text = re.sub(r'https?:\/\/\S+','', text) #Removes hyperlinks

            #not cleaning emojis as the vader sentiment analysis takes those into account 

            return text

        #Clean text and return
        combined['Tweet Text'] = combined['Tweet Text'].astype(str) #Change the tweet data type from object to string
        combined['Tweet Text'] = combined['Tweet Text'].apply(cleanText)
        combined.drop_duplicates(subset=['Tweet Text'], inplace=True)
        combined.reset_index(inplace=False)
        combined = combined.rename(columns={"Tweet Text": "Tweet", "Date & Time" : "Date"})
        # combined.to_gbq("is3107-project-383009.Dataset.tslaTweetsRealTime2", project_id="is3107-project-383009")

        print(combined)

        def sentiment_score_transform(df):
            # df['Tweet Text'] = df['Tweet Text'].astype(str)
            # df['Date & Time'] = df['Date & Time'].astype(str)
            df.dropna(subset=['Tweet', 'Date'], inplace=True)
            df["Tweet"] = df["Tweet"].str.strip()
            df.reset_index(inplace=False)
            df["Date"] = pd.to_datetime(df["Date"])
            sia = SentimentIntensityAnalyzer()
            df['Sentiments'] = df['Tweet'].apply(lambda Tweet: sia.polarity_scores(Tweet))
            df = pd.concat([df.drop(['Sentiments'], axis=1), df['Sentiments'].apply(pd.Series)], axis=1)
            print(df)
            df.to_gbq("is3107-project-383009.Dataset.realTimeAnalysed999", project_id="is3107-project-383009", if_exists='replace')
            print("successfully LOADED!")

        sentiment_score_transform(combined)








## EXTRACTION FROM YFINANCE 

def download_stock_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=15*365)

    history = yf.download("tsla", start=start_date, end=end_date, interval='1d', prepost=False)
    history = history.loc[:, ['Open', 'Close', 'Volume']]
    history.reset_index(inplace=True)
    history.to_gbq("is3107-project-383009.Dataset.tslaStock4", project_id="is3107-project-383009")

############################################
#2. DEFINE AIRFLOW DAG (SETTINGS + SCHEDULE)
############################################


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 4, 17),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'tsla_tweets_dag',
    default_args=default_args,
    catchup=False,
    schedule_interval='@daily'
)


# ##########################################
# #3. DEFINE AIRFLOW OPERATORS
# ##########################################


with dag:
    process_tweets_task = PythonOperator(
        task_id='process_tweets',
        python_callable=process_tweets,
    )

    download_stock_data_task = PythonOperator(
        task_id='download_stock_data',
        python_callable=download_stock_data,
    )

process_tweets_task 
download_stock_data_task


