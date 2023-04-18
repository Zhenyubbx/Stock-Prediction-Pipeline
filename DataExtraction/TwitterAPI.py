import requests
from datetime import datetime, timedelta
import pandas as pd
import re
from langdetect import detect

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

def cleanText(text):

  lang = detect(text) 

  if lang == 'en': 
    text = re.sub(r'@[A-Za-z0-9]+', '', text) #Removes @mentions
    text = re.sub(r'#', '', text) #Removes the # symbol
    text = re.sub(r'RT[\s]+', '', text) #Removes RT
    text = re.sub(r'https?:\/\/\S+','', text) #Removes hyperlinks
    #not cleaning emojis as the vader sentiment analysis takes those into account 
    return text
  else:
    return None #Removes non-english tweets

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
            date = datetime.strptime(str(tweet['created_at'].replace("Z", "")), '%Y-%m-%dT%H:%M:%S.000')
            data_list.append({
                'Date': date.strftime('%Y-%m-%d'),
                'User': tweet['author_id'],
                'Tweet': tweet['text']
            })
            
        for tweet in json_response_16['data']:
            date = datetime.strptime(str(tweet['created_at'].replace("Z", "")), '%Y-%m-%dT%H:%M:%S.000')
            data_list.append({
                'Date': date.strftime('%Y-%m-%d'),
                'User': tweet['author_id'],
                'Tweet': tweet['text']
            })

        combined = pd.DataFrame(data=data_list)

        #Clean text and return
        combined['Tweet'] = combined['Tweet'].astype(str) #Change the tweet data type from object to string
        combined['Tweet'] = combined['Tweet'].apply(cleanText)
        print(combined)
        print(type(combined['Date'][0]))
        combined.to_gbq("is3107-project-383009.Dataset.tslaTweets1617Apr", project_id="is3107-project-383009")
    
    # with open('tsla_tweets.json') as file:
    #     print(file.read())
            
    # print("JSON file successfully written.")


if __name__ == "__main__":
    main()
