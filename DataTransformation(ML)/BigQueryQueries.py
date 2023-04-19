from google.cloud import bigquery
from google.oauth2 import service_account

def queryStockFromBigQuery():
    CREDS = './is3107-project-383009-f4cdb3dfb0cb.json'
    credentials = service_account.Credentials.from_service_account_file(CREDS)
    client = bigquery.Client(credentials=credentials)
    job_config = bigquery.QueryJobConfig()

    # Set the destination table
    table = client.dataset(dataset_id='is3107-project-383009.Dataset').table(table_id="is3107-project-383009.Dataset.tslaStock")
    job_config.destination = table
    query = "SELECT * FROM `is3107-project-383009.Dataset.tslaStock`"
    return client.query(query).to_dataframe()

def queryTweetsFromBigQuery():
    CREDS = 'is3107-project-383009-f4cdb3dfb0cb.json'
    credentials = service_account.Credentials.from_service_account_file(CREDS)
    client = bigquery.Client(credentials=credentials)
    job_config = bigquery.QueryJobConfig()

    # Set the destination table
    table = client.dataset(dataset_id='is3107-project-383009.Dataset').table(table_id="is3107-project-383009.Dataset.tslaTweets1617Apr")
    job_config.destination = table
    query = "SELECT * FROM `is3107-project-383009.Dataset.tslaTweetsKaggle`"
    return client.query(query).to_dataframe()

def queryTweetsWithSentimentScoresFromBigQuery():
    CREDS = 'is3107-project-383009-f4cdb3dfb0cb.json'
    credentials = service_account.Credentials.from_service_account_file(CREDS)
    client = bigquery.Client(credentials=credentials)
    job_config = bigquery.QueryJobConfig()

    # Set the destination table
    table = client.dataset(dataset_id='is3107-project-383009.Dataset').table(table_id="is3107-project-383009.Dataset.tslaTweets1617Apr")
    job_config.destination = table
    query = "SELECT * FROM `is3107-project-383009.Dataset.tslaStockAnalysed`"
    return client.query(query).to_dataframe()

def queryTslaStockTweetSentimentsFromBigQuery():
    CREDS = 'is3107-project-383009-f4cdb3dfb0cb.json'
    credentials = service_account.Credentials.from_service_account_file(CREDS)
    client = bigquery.Client(credentials=credentials)
    job_config = bigquery.QueryJobConfig()

    # Set the destination table
    table = client.dataset(dataset_id='is3107-project-383009.Dataset').table(table_id="is3107-project-383009.Dataset.tslaStockTweetSentiments")
    job_config.destination = table
    query = "SELECT * FROM `is3107-project-383009.Dataset.tslaStockTweetSentiments`"
    return client.query(query).to_dataframe()