import requests
import pandas as pd
import configparser
# from google.cloud import bigquery


url = "https://yh-finance.p.rapidapi.com/stock/v3/get-historical-data"

querystring = {"symbol":"meta","region":"US"} #Currently reading Meta stock data

config = configparser.ConfigParser()
config.read('config.ini')

headers = {
	"X-RapidAPI-Key": f"{config['yhFinance']['RAPID_API_KEY']}",
	"X-RapidAPI-Host": "yh-finance.p.rapidapi.com"
}

response = requests.request("GET", url, headers=headers, params=querystring)

responseJson = response.json()
filtered_data = []
for day in responseJson["prices"]:
    day_data = {
                'date': day['date'], #Date in number of seconds since epoch
                'open': day['open'],
                'close': day['close'],
                'volume': day['volume'],
                }
    filtered_data.append(day_data)

df = pd.DataFrame(data=filtered_data)
df.to_csv("YhFinance.csv", index=False)
df.to_gbq("is3107-project-383009.Dataset.test2", project_id="is3107-project-383009")

# Alternative way to push data to BigQuery

# client = bigquery.Client(project="is3107-project-383009")
# dataset_ref = client.dataset(dataset_id="is3107-project-383009.Dataset")
# table_ref = dataset_ref.table(table_id="is3107-project-383009.Dataset.YhFinance")
# loadjob = client.load_table_from_dataframe(df, table_ref).result()
# print("Loadjob ID" + loadjob)