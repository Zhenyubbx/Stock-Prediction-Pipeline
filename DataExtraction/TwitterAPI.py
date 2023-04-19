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

query_params = {'query': '#tsla', 'max_results': '100', 'tweet.fields': 'created_at,author_id'}

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
    tweets = []

    
    for page in range(10):
        query_params['next_token'] = None
        json_response = connect_to_endpoint(search_url, query_params)
        tweets += json_response['data']
        print(f"Successfully pulled {len(json_response['data'])} tweets from Tweepy! Page {page+1}")

        if 'next_token' in json_response['meta']:
            query_params['next_token'] = json_response['meta']['next_token']
        else:
            break

    with open('tsla_tweets.json', mode='w') as file:
        data_list = []

        for tweet in tweets:
            date = datetime.strptime(str(tweet['created_at'].replace("Z", "")), '%Y-%m-%dT%H:%M:%S.000')
            data_list.append({
                'Date': str(date.strftime('%Y-%m-%d')),
                'Tweet': tweet['text']
            })

        combined = pd.DataFrame(data=data_list)

        #Clean text and return
        combined['Tweet'] = combined['Tweet'].astype(str) #Change the tweet data type from object to string
        combined['Tweet'] = combined['Tweet'].apply(cleanText)
        print(combined)
        combined.to_gbq("is3107-project-383009.Dataset.tslaTweetsRealTime", project_id="is3107-project-383009")
    
    # with open('tsla_tweets.json') as file:
    #     print(file.read())
            
    # print("JSON file successfully written.")


if __name__ == "__main__":
    main()
