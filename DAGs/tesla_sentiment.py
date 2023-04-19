from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
from textblob import TextBlob
from langdetect import detect
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *

def load_csv():
    df = pd.read_csv("/Users/darryl/airflow/dags/tesla_kaggle.csv")
    df = df[["Date & Time", "Tweet Text"]]
    print("Successfully loaded!")
    return df

def filter_valid_dates(date_str):
    try:
        datetime.strptime(date_str, '%B %d, %Y at %I:%M%p')
        return True
    except ValueError:
        return False

def clean_data():
    df = load_csv()
    valid_dates_mask = df['Date & Time'].apply(filter_valid_dates)
    df = df[valid_dates_mask]
    print(df["Date & Time"])
    df["Date & Time"] = pd.to_datetime(df["Date & Time"], format="%B %d, %Y at %I:%M%p")
    df["Date & Time"] = df["Date & Time"].dt.strftime("%Y-%m-%d")
    df["Tweet Text"] = df["Tweet Text"].str.replace(r"http\S+", "")
    df["Tweet Text"] = df["Tweet Text"].str.replace(r"@\S+", "")
    df["Tweet Text"] = df["Tweet Text"].str.replace(r"#\S+", "")
    df["Tweet Text"] = df["Tweet Text"].str.strip()
    df.to_csv("cleaned_hashtag_tesla_TweetTextsTEST.csv", index=False)
    print("successfully cleaned!")
    print(type(df))
    return df


def sentiment_score_transform(data):
    df = pd.read_csv("/Users/darryl/cleaned_hashtag_tesla_TweetTextsTEST.csv")
    print(type(df))
    df['Tweet Text'] = df['Tweet Text'].astype(str)
    df['Date & Time'] = df['Date & Time'].astype(str)
    df.dropna(subset=['Tweet Text', 'Date & Time'], inplace=False)
    df["Tweet Text"] = df["Tweet Text"].str.strip()
    df.reset_index(inplace=False)
    sia = SentimentIntensityAnalyzer()
    df['Sentiments'] = df['Tweet Text'].apply(lambda Tweet: sia.polarity_scores(Tweet))
    df = pd.concat([df.drop(['Sentiments'], axis=1), df['Sentiments'].apply(pd.Series)], axis=1)
    df.to_gbq("is3107-project-383009.Dataset.tslaStockAnalysed2", project_id="is3107-project-383009")
    print("successfully LOADED!")


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 4, 19),
    'depends_on_past': False,
    'catchup': False,
}
dag = DAG('tesla_sentiment_analysis',
          default_args=default_args,
          description='This DAG loads and cleans Tesla tweet data, performs sentiment analysis on the text, and stores the results in a BigQuery table.'
          )


with DAG(dag_id='tesla_sentiment_analysis',
         default_args=default_args,
         schedule_interval=None) as dag:

    t1 = PythonOperator(
        task_id='load_csv',
        python_callable=load_csv
    )

    t2 = PythonOperator(
        task_id='clean_data',
        python_callable=clean_data,
        op_kwargs={'return_value': True},
        provide_context=True
    )

    t3 = PythonOperator(
        task_id='sentiment_score_transform',
        python_callable=sentiment_score_transform,
        op_kwargs={'data': '{{ task_instance.xcom_pull(task_ids="clean_data") }}'}
    )


t1 >> t2 >> t3