# import tweepy
# from textblob import TextBlob
# from langdetect import detect
# from nltk.classify import NaiveBayesClassifier
# from nltk.corpus import subjectivity
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
import pandas as pd
# import numpy as np
# import re
import matplotlib.pyplot as plt
# import nltk
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# from treeinterpreter import treeinterpreter as ti

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

def sentiment_score_transform(data):
    # data['Tweet'] = data['Tweet'].astype(str) #Change the tweet data type from object to string
    data.dropna(subset=['Tweet'], inplace=True) #drop rows with na values in Tweet column
    data.reset_index(drop=True, inplace=True) #reset indexes and drop the old index column
    sia = SentimentIntensityAnalyzer()
    data['Sentiments'] = data['Tweet'].apply(lambda Tweet: sia.polarity_scores(Tweet))
    data = pd.concat([data.drop(['Sentiments'], axis=1), data['Sentiments'].apply(pd.Series)], axis=1)
    # data.to_gbq("is3107-project-383009.Dataset.tslaStockAnalysed", project_id="is3107-project-383009")
    # print(data)
    return data

def transform_df(df):
    df = df.dropna()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by='Date', ascending=False)
    sum_by_date = df.groupby('Date')['compound'].mean()
    Close = df.groupby('Date')['Close'].mean()
    new_df = pd.concat([sum_by_date, Close], axis=1)
    new_df.reset_index(inplace=True)
    new_df["Ytd_Close"] = new_df["Close"].shift(1)
    new_df["Ytd_Compound"] = new_df["compound"].shift(1)
    new_df = new_df.dropna()
    return new_df

def get_data():
    tweetsDf = queryTweetsFromBigQuery()
    tweetsWithSentimentScores = sentiment_score_transform(tweetsDf)
    # tweetsWithSentimentScores = queryTweetsWithSentimentScoresFromBigQuery()
    stockDf = queryStockFromBigQuery()
    #Combining dataframes
    combined = tweetsWithSentimentScores.merge(stockDf, on='Date')
    # combined.to_csv('combined.csv', index=False)
    return combined

def use_linear_model(combined):
    # combined = pd.read_csv("combined.csv")
    new_df = transform_df(combined)

    train, test = train_test_split(new_df, shuffle=False, test_size=0.2)
    train_x = train[["Ytd_Compound", "Ytd_Close"]]
    train_y = train[["Close"]]
    linear_model_Regr = linear_model.LinearRegression()
    linear_model_Regr.fit(train_x, train_y)

    test_x = test[["Ytd_Compound", "Ytd_Close"]]
    test_y = test[["Close"]]

    y_pred = linear_model_Regr.predict(test_x)
    mse = mean_squared_error(test_y, y_pred)

    print("Mean Squared Error: {:.2f}".format(mse))
    plot_graph(test['Date'], y_pred, test_y, 'Linear Model')
    
    return linear_model_Regr


def use_random_forest_regressor(combined):
    # combined = pd.read_csv("combined.csv")
    new_df = transform_df(combined)

    train, test = train_test_split(new_df, shuffle=False, test_size=0.2)
    train_x = train[["Ytd_Compound", "Ytd_Close"]]
    train_y = train[["Close"]]

    test_x = test[["Ytd_Compound", "Ytd_Close"]]
    test_y = test[["Close"]]

    rfg = RandomForestRegressor()
    rfg.fit(train_x, train_y)
    y_pred = rfg.predict(test_x)
    mse = mean_squared_error(test_y, y_pred)

    print("Mean Squared Error: {:.2f}".format(mse))
    plot_graph(test['Date'], y_pred, test_y, "Random Forest Regressor")


def plot_graph(dates, predicted_prices, actual_prices, model):
    # create a figure and axis object
    fig, ax = plt.subplots()

    # plotting
    ax.plot(dates, actual_prices, color='blue', label='Actual Closing Prices')
    ax.plot(dates, predicted_prices, color='red', label= 'Predicted Closing Prices')
    ax.set_xlabel('Date')
    ax.set_ylabel('Prices/$')
    ax.set_title(f'Prediction using {model}')

    # create a second axis object with a different scale for y2
    # ax2 = ax.twinx()
    # ax2.plot(test['Date'], y_pred, color='red')
    # ax2.set_ylabel('Predicted Closing Prices')

    # display the plot
    ax.legend()
    plt.show()

def predict_with_linear_model(tdy_closing_price, tdy_compound):
    data = get_data()
    model = use_linear_model(data)
    predicted_price = model.predict([[tdy_closing_price,tdy_compound]]) #Predict today's closing price with ytd's closing price and ytd's compound
    return predicted_price

def test_predict_with_linear_model():
    return predict_with_linear_model(0.095435,195.970001)[0][0]




# print(test_predict_with_linear_model())

