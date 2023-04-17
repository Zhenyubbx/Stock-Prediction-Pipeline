import requests
import os
import json
import csv
from datetime import datetime, timedelta

current_time = datetime.utcnow()
ten_seconds = timedelta(seconds=10)
end_time = current_time - ten_seconds



# To set your environment variables in your terminal run the following line:
# export 'BEARER_TOKEN'='<your_bearer_token>'
bearer_token = "AAAAAAAAAAAAAAAAAAAAACYhmwEAAAAAl7HpRkqggtVjTCj0LmMI1nMYaQg%3Dp6OKovTzZ4XI6GuDfiscptdLvv35yp9LZNnkTShVYTwblI6ARa"

search_url = "https://api.twitter.com/2/tweets/search/recent"

# Optional params: start_time,end_time,since_id,until_id,max_results,next_token,
# expansions,tweet.fields,media.fields,poll.fields,place.fields,user.fields
query_params_17 = {'query': '#tsla', 'max_results': '100', 'tweet.fields': 'created_at,author_id',
                   'start_time': '2023-04-17T00:00:00Z', 'end_time': end_time.strftime('%Y-%m-%dT%H:%M:%SZ')}


query_params_16 = {'query': '#tsla', 'max_results': '100', 'tweet.fields': 'created_at,author_id',
                   'start_time': '2023-04-16T00:00:00Z', 'end_time': '2023-04-16T23:59:59Z'}

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


def main():
    # Get 20 tweets from April 17
    json_response_17 = connect_to_endpoint(search_url, query_params_17)
    tweets_17 = json_response_17['data']

    # Get 20 tweets from April 16
    json_response_16 = connect_to_endpoint(search_url, query_params_16)
    tweets_16 = json_response_16['data']
    
    with open('tsla_tweets.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', 'User', 'Tweet'])
        
        for tweet in tweets_17:
            created_at = tweet['created_at']
            author_name = tweet['author_id']
            text = tweet['text']
            writer.writerow([created_at, author_name, text])
            
        for tweet in tweets_16:
            created_at = tweet['created_at']
            author_name = tweet['author_id']
            text = tweet['text']
            writer.writerow([created_at, author_name, text])
            
    print("CSV file successfully written.")

if __name__ == "__main__":
    main()
