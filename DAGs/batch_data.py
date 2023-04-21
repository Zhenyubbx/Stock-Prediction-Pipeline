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

def extraction_of_tweets():
    ## change line 14 to your path directory to read csv.
    df = pd.read_csv("/Users/darryl/airflow/dags/stock_tweets.csv")
    df = df[["Date", "Tweet", "Stock Name"]]
    df = df.rename(columns={"Date": "Date", "Tweet" : "Tweet", "Stock Name": "StockName"})
    print("Successfully extracted batch data!")
    return df

def filter_valid_dates(date_str):
    try:
        datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S%z')
        return True
    except ValueError:
        return False

def clean_data():
    df = extraction_of_tweets()
    valid_dates_mask = df['Date'].apply(filter_valid_dates)
    df = df[valid_dates_mask]
    print(df["Date"])
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d %H:%M:%S%z")
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df["Tweet"] = df["Tweet"].str.replace(r"http\S+", "")
    df["Tweet"] = df["Tweet"].str.replace(r"@\S+", "")
    df["Tweet"] = df["Tweet"].str.replace(r"#\S+", "")
    df["Tweet"] = df["Tweet"].str.strip()
    print("Successfully cleaned!")
    df.dropna()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by='Date', ascending=False)
    df.reset_index(inplace=False)
    # df.to_csv("/Users/darryl/airflow/dags/allTweetsKaggleCleaned.csv", index=False)
    df.to_gbq("is3107-project-383009.Dataset.allStockTweetsCleaned", project_id="is3107-project-383009", if_exists='replace')
    print("Successfully loaded cleaned data into GBQ!")
    return df


def transform(**context):
    data = context['ti'].xcom_pull(task_ids='clean_data')

#    data = pd.read_csv("allTweetsKaggleCleaned.csv")
    data.dropna(subset=['Tweet'], inplace=True) #drop rows with na values in Tweet column
    data.reset_index(drop=True, inplace=True) #reset indexes and drop the old index column
    sia = SentimentIntensityAnalyzer()
    data['Sentiments'] = data['Tweet'].apply(lambda Tweet: sia.polarity_scores(Tweet))
    data = pd.concat([data.drop(['Sentiments'], axis=1), data['Sentiments'].apply(pd.Series)], axis=1)
    print("Sentiment analysed and scored!")
    # print(data)
    return data

def load(**context):
    data = context['ti'].xcom_pull(task_ids='transform')
    data.to_gbq("is3107-project-383009.Dataset.kaggleSentimentAnalysed", project_id="is3107-project-383009", if_exists='replace')
    return data



    
    


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 4, 19),
    'depends_on_past': False,
    'catchup': False,
}
dag = DAG('batch_data_dag',
          default_args=default_args,
          description='This DAG loads and cleans kaggle batch tweets data, performs sentiment analysis on the text, and stores the results in a BigQuery table.'
          )


with DAG(dag_id='batch_data_dag',
         default_args=default_args,
         schedule_interval=None) as dag:

    t1 = PythonOperator(
        task_id='extraction_of_tweets',
        python_callable=extraction_of_tweets
    )

    t2 = PythonOperator(
        task_id='clean_data',
        python_callable=clean_data,
        provide_context=True
    )

    t3 = PythonOperator(
        task_id='transform',
        python_callable=transform,
        provide_context=True
    )

    t4 = PythonOperator(
        task_id='load',
        python_callable=load,
        provide_context=True
    )

    


t1 >> t2 >> t3 >> t4