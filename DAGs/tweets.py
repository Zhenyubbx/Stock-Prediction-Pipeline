import requests
import os
import json
import csv
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import yfinance as yf
import pandas as pd

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
    query_params_17 = {'query': '#tsla', 'max_results': '100', 'tweet.fields': 'created_at,author_id',
                       'start_time': '2023-04-17T00:00:00Z', 'end_time': end_time.strftime('%Y-%m-%dT%H:%M:%SZ')}

    query_params_16 = {'query': '#tsla', 'max_results': '100', 'tweet.fields': 'created_at,author_id',
                       'start_time': '2023-04-16T00:00:00Z', 'end_time': '2023-04-16T23:59:59Z'}

    # Get 100 tweets from April 17
    json_response_17 = connect_to_endpoint(search_url, query_params_17)
    tweets_17 = json_response_17['data']

    # Get 100 tweets from April 16
    json_response_16 = connect_to_endpoint(search_url, query_params_16)
    tweets_16 = json_response_16['data']

    tweets = tweets_17 + tweets_16
    json_response = {'tweets': tweets}
    
    print(json.dumps(json_response, indent=4))




## EXTRACTION FROM YFINANCE 

def download_stock_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=15*365)

    history = yf.download("tsla", start=start_date, end=end_date, interval='1d', prepost=False)
    history = history.loc[:, ['Open', 'Close', 'Volume']]
    history.reset_index(inplace=True)
    history.to_gbq("is3107-project-383009.Dataset.tslaStock", project_id="is3107-project-383009")

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


##########################################
#3. DEFINE AIRFLOW OPERATORS
##########################################


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




# Task 1: process and clean tweets
# Task 2: push to big query
# Task 3: take from bigquery in json format
# Task 4: twitter sentiment analysis
# Task 5: machine learning
# Task 6:  




