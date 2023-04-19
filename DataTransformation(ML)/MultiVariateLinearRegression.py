import pandas as pd
from sklearn import linear_model
from google.cloud import bigquery
from google.oauth2 import service_account
import datetime

def queryStockFromBigQuery():
    CREDS = './is3107-project-383009-f4cdb3dfb0cb.json'
    credentials = service_account.Credentials.from_service_account_file(CREDS)
    client = bigquery.Client(credentials=credentials)
    job_config = bigquery.QueryJobConfig()

    # Set the destination table
    table = client.dataset(dataset_id='is3107-project-383009.Dataset').table(table_id="is3107-project-383009.Dataset.tslaStock")
    job_config.destination = table
    query = "SELECT * FROM `is3107-project-383009.Dataset.tslaStock2`"
    return client.query(query).to_dataframe()

def queryTweetsFromBigQuery():
    CREDS = './is3107-project-383009-f4cdb3dfb0cb.json'
    credentials = service_account.Credentials.from_service_account_file(CREDS)
    client = bigquery.Client(credentials=credentials)
    job_config = bigquery.QueryJobConfig()

    # Set the destination table
    table = client.dataset(dataset_id='is3107-project-383009.Dataset').table(table_id="is3107-project-383009.Dataset.tslaTweets1617Apr")
    job_config.destination = table
    query = "SELECT * FROM `is3107-project-383009.Dataset.tslaTweetsKaggle`"
    return client.query(query).to_dataframe()

def train(stockDf, TweetsDf):
    #Combining dataframes
    combined = stockDf.merge(TweetsDf, on='Date')
    print(combined)
    # regr = linear_model.LinearRegression()
    # regr.fit(X, y)
    # predicted_price = regr.predict([[1681479000, 220.35000610351562]]) #Predict next day closing price with today's 
    # print(predicted_price)

def main():  
    stockDf = queryStockFromBigQuery()
    tweetsDf = queryTweetsFromBigQuery()
    train(stockDf, tweetsDf)
    # print(stockDf.head(5))

main()
# added_data = []
# for idx, x in enumerate(filtered_data):
#     if (idx!= 0):      
#         idx_data = {
#                     "sentimentscore": x["sentimentscore"],
#                     "close": x["close"],
#                     "ytd_close": filtered_data[idx-1]["close"],
#                     }
#         added_data.append(idx_data)

