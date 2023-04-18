import requests
import os
import json
from datetime import datetime, timedelta
import pandas as pd

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
                   'start_time': '2023-04-17T00:00:00Z', 'end_time': '2023-04-17T23:59:59Z'} # end_time.strftime('%Y-%m-%dT%H:%M:%SZ')


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
    # Get 100 tweets from April 16
    json_response_16 = connect_to_endpoint(search_url, query_params_16)
    print("Successfully pulled " + str(len(json_response_16['data'])) + " from Tweepy!")
    
    # Get 99 tweets from April 17
    json_response_17 = connect_to_endpoint(search_url, query_params_17)
    print("Successfully pulled " + str(len(json_response_17['data'])) + " from Tweepy!")
    
    with open('tsla_tweets.json', mode='w') as file:
        data_list = []
        
        for tweet in json_response_17['data']:
            data_list.append({
                'Time': tweet['created_at'],
                'User': tweet['author_id'],
                'Tweet': tweet['text']
            })
            
        for tweet in json_response_16['data']:
            data_list.append({
                'Time': tweet['created_at'],
                'User': tweet['author_id'],
                'Tweet': tweet['text']
            })
        
        json.dump(data_list, file, indent=2)
        combined = pd.DataFrame(data=data_list)
        combined.to_gbq("is3107-project-383009.Dataset.tslaTweets1617Apr", project_id="is3107-project-383009")
        print(combined)
            
    with open('tsla_tweets.json') as file:
        print(file.read())
            
    print("JSON file successfully written.")


if __name__ == "__main__":
    main()
