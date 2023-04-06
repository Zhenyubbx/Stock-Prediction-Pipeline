import tweepy
import configparser
import pandas as pd
from datetime import datetime, timedelta

config = configparser.ConfigParser()
config.read('config.ini')

api_key = config['twitter']['API_KEY']
api_key_secret = config['twitter']['API_KEY_SECRET']
access_token = config['twitter']['ACCESS_TOKEN']
access_token_secret = config['twitter']['ACCESS_TOKEN_SECRET']

# authentication
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# search tweets from two separate days
keywords = '#tsla'
date1 = datetime(2023, 4, 1) # YYYY-MM-DD
date2 = datetime(2023, 4, 2) # YYYY-MM-DD
dates = [date1, date2]
limit = 300

# create empty list to store dataframes
dfs = []

for date in dates:
    date_since = date.strftime('%Y-%m-%d')
    date_until = (date + timedelta(days=1)).strftime('%Y-%m-%d')

    tweets = tweepy.Cursor(api.search_tweets,
                           q=keywords,
                           count=20,
                           tweet_mode='extended',
                           since=date_since,
                           until=date_until).items(limit)

    # create DataFrame
    columns = ['Time', 'User', 'Tweet']
    data = []

    for tweet in tweets:
        if hasattr(tweet, 'extended_tweet'):
            text = tweet.extended_tweet.full_text
        else:
            text = tweet.full_text

        data.append([tweet.created_at, tweet.user.screen_name, text])


    df = pd.DataFrame(data, columns=columns)
    dfs.append(df)


# concatenate dataframes
result = pd.concat(dfs)

result.to_csv('tweetsfromdays.csv')
