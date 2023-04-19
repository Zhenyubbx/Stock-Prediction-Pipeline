import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
# import yfinance as yfi
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math
from sklearn.ensemble import RandomForestRegressor
from sentiment import queryTweetsWithSentimentScoresFromBigQuery, transform_df

# Set up a service account key
key_path = './is3107-project-383009-8646ee4b721b.json'
credentials = service_account.Credentials.from_service_account_file(key_path)

# Set up a client object for BigQuery using the credentials
client = bigquery.Client(credentials=credentials)

# Construct a table reference to the table you want to query
table_ref = client.get_table('is3107-project-383009.Dataset.tslaStockTweetSentiments')

# Construct a SQL query
query = """
    SELECT *
    FROM `is3107-project-383009.Dataset.tslaStockTweetSentiments`
"""

# Execute the query and get the results
df = client.query(query).to_dataframe()


# Print the first 10 rows of the DataFrame
# print(df.tail(10))

# Abbrevation for the stocks in S&P500
tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S&P_500_companies')[0]['Symbol']

print(tickers)

# Linear Regression to predict
from sklearn.linear_model import LinearRegression

# Test model
def test_it(opens, closes, preds, start_account=1000, thresh=0):
    account = start_account
    changes = []
    
    for i in range(len(preds)):
        if (preds[i]-opens[i])/opens[i] >= thresh:
            account += account*(closes[i]-opens[i])/opens[i]
        changes.append(account)
    changes = np.array(changes)
    
    # Gives graph of account over time
    plt.plot(range(len(changes)), changes)
    plt.show()
    
    invest_total = start_account + start_account*(closes[-1]-opens[0])/opens[0]
    print('Investing Total:', invest_total, str(round((invest_total-start_account)/start_account*100,1))+'%')
    print('Algo-Trading Total:', account, str(round((account-start_account)/start_account*100,1))+'%')

    
    
# MACD indicator - to indicate which way the stock is trending
def calc_macd(data, len1, len2, len3):
    shortEMA = data.ewm(span=len1, adjust=False).mean()
    longEMA = data.ewm(span=len2, adjust=False).mean()
    MACD = shortEMA - longEMA
    signal = MACD.ewm(span=len3, adjust=False).mean()
    return MACD, signal

# RSI indicator - whether a stock is overbought or oversold
def calc_rsi(data, period):
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ema_up = up.ewm(com=period, adjust=False).mean()
    ema_down = down.ewm(com=period, adjust=False).mean()
    rs = ema_up / ema_down
    rsi = 100 - (100/(1+rs))
    return rsi

# Bollinger bands - lower & upper band on price to reflect 2 std dev from the mean
def calc_bollinger(data, period):
    mean = data.rolling(period).mean()
    std = data.rolling(period).std() 
    upper_band = np.array(mean) + 2*np.array(std)
    lower_band = np.array(mean) - 2*np.array(std)
    return upper_band, lower_band

def plot_graph(dates, predicted_prices, actual_prices, model):
    # create a figure and axis object
    fig, ax = plt.subplots()

    # plotting
    ax.plot(dates, actual_prices, color='blue', label='Actual Closing Prices')
    ax.plot(dates, predicted_prices, color='red', label= 'Predicted Closing Prices')
    ax.set_xlabel('Date')
    ax.set_ylabel('Prices/$')
    ax.set_title(f'Prediction using Single Variate {model}')

    # create a second axis object with a different scale for y2
    # ax2 = ax.twinx()
    # ax2.plot(test['Date'], y_pred, color='red')
    # ax2.set_ylabel('Predicted Closing Prices')

    # display the plot
    ax.legend()
    plt.show()

def use_linear_model(df):
    for i in range(1):
        history = df
        
        # Set closing price and volume of day before
        history['Prev_Close'] = history['Close'].shift(1)
        history['Prev_Volume'] = history['Volume'].shift(1)

        # # Pull day of week
        # history = history.set_index('Date')
        # datetimes = history.index.values
        # weekdays = []
        
        # for dt in datetimes:
        #     # Parse the string into a datetime object with timezone information
        #     dt = datetime.strptime(str(dt)[:26], '%Y-%m-%dT%H:%M:%S.%f')
        
        
        # SMA indicator
        history['5SMA'] = history['Prev_Close'].rolling(5).mean()
        history['10SMA'] = history['Prev_Close'].rolling(10).mean()
        history['20SMA'] = history['Prev_Close'].rolling(20).mean()
        history['50SMA'] = history['Prev_Close'].rolling(50).mean()
        history['100SMA'] = history['Prev_Close'].rolling(100).mean()
        history['200SMA'] = history['Prev_Close'].rolling(200).mean()
        
        # MACD indicator
        MACD, signal = calc_macd(history['Prev_Close'], 12, 26, 9)
        history['MACD'] = MACD
        history['MACD_signal'] = signal
        
        # RSI indicator
        history['RSI'] = calc_rsi(history['Prev_Close'], 13)
        history['RSI_Volume'] = calc_rsi(history['Prev_Volume'], 13)
        
        # Bollinger band indicator
        upper, lower = calc_bollinger(history['Prev_Close'], 20)
        history['Upper_Band'] = upper
        history['Lower_Band'] = lower
        
        
        labels = ['Prev_Close', 'Prev_Volume', '5SMA', '10SMA', '20SMA', '50SMA', '100SMA', '200SMA', 'MACD', 'MACD_signal', 'RSI', 'RSI_Volume', 'Upper_Band', 'Lower_Band']
        
        # 1 day percentage change
        period = 1
        new_labels = [str(period)+'d_'+label for label in labels]
        history[new_labels] = history[labels].pct_change(period, fill_method='ffill')
        
        # 2 day percentage change
        period = 2
        new_labels = [str(period)+'d_'+label for label in labels]
        history[new_labels] = history[labels].pct_change(period, fill_method='ffill')
        
        # 5 day percentage change
        period = 5
        new_labels = [str(period)+'d_'+label for label in labels]
        history[new_labels] = history[labels].pct_change(period, fill_method='ffill')
        
        # 10 day percentage change
        period = 10
        new_labels = [str(period)+'d_'+label for label in labels]
        history[new_labels] = history[labels].pct_change(period, fill_method='ffill')
        
        # Clean empty data 
        history = history.replace(np.inf, np.nan).dropna()
        
        # Get data
        # y = history['Close']
        # X = history.drop(['Close', 'Volume'], axis=1).values

        # Split into training & testing data
        # num_test = 10
        # X_train = X[:-1*num_test]
        # y_train = y[:-1*num_test]
        # X_test = X[-1*num_test:]
        # y_test = y[-1*num_test:]
        df = transform_df(df)
        print(df)
        train, test = train_test_split(df, shuffle=False, test_size=0.2)
        X_train = train[["Ytd_Close"]]
        y_train = train[["Close"]]

        # Define model and make predictions
        model = LinearRegression()
        model = model.fit(X_train, y_train)

        X_test = test[["Ytd_Close"]]
        test_y = test[["Close"]]
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(test_y, y_pred)
        rmse = math.sqrt(mse)

        print("Root Mean Squared Error: {:.2f}".format(rmse))
        plot_graph(test['Date'], y_pred, test_y, 'Linear Model')
        # result = pd.DataFrame(preds, columns=['Predicted price'])
        # for i in range(len(result['Predicted price'])):
        #     result['Predicted price'][i] = float(result['Predicted price'][i])

        
        # result.to_gbq("is3107-project-383009.Dataset.regressionResults", project_id="is3107-project-383009", if_exists='replace')
        
        
    #     test_it(X_test.T[0], y_test, preds, 1000, 0.005)
        
        
        
    #     x = history.index.values
        
    #     # Display SMA & Bollinger Band
    #     plt.figure(figsize=(15, 5))
    #     plt.plot(x, history['Prev_Close'], color='blue')
    #     plt.plot(x, history['50SMA'], color='green')
    #     plt.plot(x, history['200SMA'], color='red')
    #     plt.plot(x, history['Upper_Band'], color='orange')
    #     plt.plot(x, history['Lower_Band'], color='orange')
    #     plt.xlim(x[1000], x[1500])
    #     plt.show()
        
    #     # Display MACD
    #     plt.figure(figsize=(15, 3))
    #     colors = np.array(['green']*len(history['MACD']))
    #     colors[history['MACD'] < 0] = 'red'
    #     plt.bar(x, history['MACD'], color=colors)
    #     plt.plot(x, history['MACD_signal'], color='blue')
    #     plt.xlim(x[1000], x[1500])
    #     plt.show()
        
    #     # Display RSI
    #     plt.figure(figsize=(15,3))
    #     plt.plot(x, history['RSI'], color='purple')
    #     plt.plot([x[0], x[-1]], [80,80], color='red')
    #     plt.plot([x[0], x[-1]], [20,20], color='green')
    #     plt.xlim(x[1000], x[1500])
    #     plt.ylim(0, 100)
    #     plt.show()
def use_random_forest_regressor(df):
    for i in range(1):
        df = transform_df(df)
        print(df)
        train, test = train_test_split(df, shuffle=False, test_size=0.2)
        X_train = train[["Ytd_Close"]]
        y_train = train[["Close"]]

        # Define model and make predictions
        model = RandomForestRegressor()
        model = model.fit(X_train, y_train)

        X_test = test[["Ytd_Close"]]
        test_y = test[["Close"]]
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(test_y, y_pred)
        rmse = math.sqrt(mse)

        print("Root Mean Squared Error: {:.2f}".format(rmse))
        plot_graph(test['Date'], y_pred, test_y, 'Random Forest Regressor')

use_random_forest_regressor(df)