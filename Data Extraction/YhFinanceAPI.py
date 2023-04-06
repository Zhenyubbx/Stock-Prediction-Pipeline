import requests
import os
from dotenv import load_dotenv
import configparser

url = "https://yh-finance.p.rapidapi.com/stock/v3/get-historical-data"

querystring = {"symbol":"meta","region":"US"}

load_dotenv()

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
                "date": day["date"], #Date in number of seconds since epoch
                "close": day['close'],
                }
    filtered_data.append(day_data)

print(filtered_data)