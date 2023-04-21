import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# from treeinterpreter import treeinterpreter as ti
import math
from BigQueryQueries import query_table_from_bq
import time

def get_data():
    stockData = query_table_from_bq("StockData")
    print("Successfully queried allStockTweetsCleaned")
    kaggleSentimentAnalysed = query_table_from_bq("kaggleSentimentAnalysed")
    print("Successfully queried kaggleSentimentAnalaysed")
    #Combining dataframes
    # combined = tweetsWithSentimentScores.merge(stockDf, on='Date')
    # combined.to_csv('combined.csv', index=False)
    return stockData, kaggleSentimentAnalysed

def transform_df(stockData, kaggleSentimentAnalysed):
    stockData = stockData.dropna()
    kaggleSentimentAnalysed = kaggleSentimentAnalysed.dropna()

    stockData = stockData.rename(columns={'Symbol':'StockName'})
    # kaggleSentimentAnalysed['Date'] = pd.to_datetime(kaggleSentimentAnalysed['Date']).dt.tz_localize('UTC')
    combined = pd.merge(stockData, kaggleSentimentAnalysed, on=['StockName', 'Date'])
    print("Successfully merged allStocksAnalysed and allStockTweetsCleaned")
    grouped = combined.groupby('StockName')

    transformed_df = pd.DataFrame()
    for StockName, group in grouped:
        # print("*****************************************************")
        # print(StockName)
        # print(group)

        group = group.sort_values(by='Date', ascending=False)
        sum_by_date = group.groupby('Date')['compound'].mean()
        Close = group.groupby('Date')['Close'].mean()
        new_df = pd.concat([sum_by_date, Close], axis=1)
        new_df.reset_index(inplace=True)
        new_df["Ytd_Close"] = new_df["Close"].shift(1)
        new_df["Ytd_Compound"] = new_df["compound"].shift(1)
        new_df = new_df.dropna()
        new_df['StockName'] = StockName
        transformed_df = pd.concat([transformed_df, new_df])
        # return new_df
        print(f"Successfully transformed {StockName} group")

    print("Successfully transformed all groups")
    return transformed_df

def use_linear_model(combined, stockName):
    # combined = pd.read_csv("combined.csv")
    # new_df = transform_df(combined)

    train, test = train_test_split(combined, shuffle=False, test_size=0.2)
    train_x = train[["Ytd_Compound"]]
    train_y = train[["Close"]]
    linear_model_Regr = LinearRegression()

    start_training_time = time.time()
    linear_model_Regr.fit(train_x, train_y)
    end_training_time = time.time()
    training_time = end_training_time - start_training_time
    print("Training time for {}: {:.2f} seconds".format(stockName, training_time))

    test_x = test[["Ytd_Compound"]]
    test_y = test[["Close"]]

    start_predict_time = time.time()
    y_pred = linear_model_Regr.predict(test_x)
    end_predict_time = time.time()
    predict_time_per_price = (end_predict_time - start_predict_time)/len(test_x.index)
    print("Predict time per price for {}: {:.2f} seconds".format(stockName, predict_time_per_price))

    test["Predicted_Price"] = y_pred
    mse = mean_squared_error(test_y, y_pred)
    rmse = math.sqrt(mse)

    print("Root Mean Squared Error for {}: {:.2f}".format(stockName, rmse))
    plot_graph(test['Date'], y_pred, test_y, 'Linear Model', stockName)

    df_with_predicted = pd.concat([train, test])
    # df_with_predicted.to_gbq("is3107-project-383009.Dataset.singleVariateLinearResults", project_id="is3107-project-383009", if_exists='replace')
    
    return df_with_predicted, training_time, predict_time_per_price, rmse

def use_random_forest_regressor(combined, stock_name):
    # combined = pd.read_csv("combined.csv")
    # new_df = transform_df(combined)

    train, test = train_test_split(combined, shuffle=False, test_size=0.2)
    train_x = train[["Ytd_Compound"]]
    train_y = train[["Close"]]

    test_x = test[["Ytd_Compound"]]
    test_y = test[["Close"]]

    rfg = RandomForestRegressor()
    # Start the timer
    start_training_time = time.time()
    rfg.fit(train_x, train_y)
    # End the timer
    end_training_time = time.time()
    training_time = end_training_time - start_training_time
    print("Training time for {}: {:.2f} seconds".format(stock_name, training_time))

    start_predict_time = time.time()
    y_pred = rfg.predict(test_x)
    end_predict_time = time.time()
    predict_time_per_price = (end_predict_time - start_predict_time)/len(test_x.index)
    print("Predict time per price for {}: {:.2f} seconds".format(stock_name, predict_time_per_price))

    test["Predicted_Price"] = y_pred
    mse = mean_squared_error(test_y, y_pred)
    rmse = math.sqrt(mse)

    print("Root Mean Squared Error for {}: {:.2f}".format(stock_name, rmse))
    # plot_graph(test['Date'], y_pred, test_y, "Random Forest Regressor", stock_name)

    df_with_predicted = pd.concat([train, test])
    # df_with_predicted.to_gbq("is3107-project-383009.Dataset.singleVariateLinearResults", project_id="is3107-project-383009", if_exists='replace')

    return df_with_predicted, training_time, predict_time_per_price, rmse

def plot_graph(dates, predicted_prices, actual_prices, model, StockName):
    # create a figure and axis object
    fig, ax = plt.subplots()

    # plotting
    ax.plot(dates, actual_prices, color='blue', label='Actual Closing Prices')
    ax.plot(dates, predicted_prices, color='red', label= 'Predicted Closing Prices')
    ax.set_xlabel('Date')
    ax.set_ylabel('Prices/$')
    ax.set_title(f'Prediction using Multi Variate {model} for {StockName} price')

    # create a second axis object with a different scale for y2
    # ax2 = ax.twinx()
    # ax2.plot(test['Date'], y_pred, color='red')
    # ax2.set_ylabel('Predicted Closing Prices')

    # display the plot
    ax.legend()
    plt.show()

def analyse_with_linear_model(transformed_df):
    grouped = transformed_df.groupby('StockName')
    analysed = pd.DataFrame()
    total_training_time = 0
    total_predict_time_per_price = 0
    total_rmse = 0
    num_of_stocks = 0
    for StockName, group in grouped:
        # print("*****************************************************")
        # print(StockName)
        # print(group) 

        df_with_predicted, training_time, predict_time_per_price, rmse = use_linear_model(group, StockName)
        print(f"Successfully trained single variate linear model for {StockName} group")
        # print(df_with_predicted.head(5))
        analysed = pd.concat([analysed, df_with_predicted])
        total_training_time = total_training_time + training_time
        total_predict_time_per_price = total_predict_time_per_price + predict_time_per_price
        total_rmse = total_rmse + rmse
        num_of_stocks = num_of_stocks +1
    average_rsme = total_rmse/num_of_stocks
    average_training_time = total_training_time/num_of_stocks
    average_predict_time_per_price = total_predict_time_per_price/num_of_stocks
    print(f"Successfully trained single variate linear model for all groups")
    print(f"Average training time: {average_training_time}")
    print(f"Average predict time per stock price: {average_predict_time_per_price}")
    print(f"Average RSME: {average_rsme}")
    print(analysed.head(5))
    analysed.to_gbq("is3107-project-383009.Dataset.singleVariateLinearResults", project_id="is3107-project-383009", if_exists='replace')


def analyse_with_random_forest(transformed_df):
    grouped = transformed_df.groupby('StockName')
    analysed = pd.DataFrame()
    total_training_time = 0
    total_predict_time_per_price = 0
    total_rmse = 0
    num_of_stocks = 0
    for StockName, group in grouped:
        # print("*****************************************************")
        # print(StockName)
        # print(group) 

        df_with_predicted, training_time, predict_time_per_price, rmse = use_random_forest_regressor(group, StockName)
        print(f"Successfully trained single variate random forest model for {StockName} group")
        # print(df_with_predicted.head(5))
        analysed = pd.concat([analysed, df_with_predicted])
        total_training_time = total_training_time + training_time
        total_predict_time_per_price = total_predict_time_per_price + predict_time_per_price
        total_rmse = total_rmse + rmse
        num_of_stocks = num_of_stocks +1
    average_rsme = total_rmse/num_of_stocks
    average_training_time = total_training_time/num_of_stocks
    average_predict_time_per_price = total_predict_time_per_price/num_of_stocks
    print(f"Successfully trained single variate random forest model for all groups")
    print(f"Average training time: {average_training_time}")
    print(f"Average predict time per stock price: {average_predict_time_per_price}")
    print(f"Average RSME: {average_rsme}")
    print(analysed.head(5))
    analysed.to_gbq("is3107-project-383009.Dataset.singleVariateRandomForestResults", project_id="is3107-project-383009", if_exists='replace')


def count(transformed_df):
    grouped = transformed_df.groupby('StockName')
    print(f"num of total stock prices: {len(transformed_df.index)}")
    num_of_stocks = 0
    for StockName, group in grouped:
        num_of_stocks = num_of_stocks + 1
    print(f"num of stocks: {num_of_stocks}")
        

# stockData, kaggleSentimentAnalysed = get_data()
# grouped = transform_df(stockData, kaggleSentimentAnalysed)
# analyse_with_linear_model(grouped)