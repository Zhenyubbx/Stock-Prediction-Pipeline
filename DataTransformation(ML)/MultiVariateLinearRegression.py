import pandas as pd
from sklearn import linear_model
from DataExtraction.YhFinanceAPI import filtered_data

added_data = []
for idx, x in enumerate(filtered_data):
    if (idx!= 0):      
        idx_data = {
                    "date": x["date"],
                    "close": x["close"],
                    "ytd_close": filtered_data[idx-1]["close"],
                    }
        added_data.append(idx_data)

df = pd.DataFrame(data=added_data)
X = df[['date', 'ytd_close']]
y = df['close']
regr = linear_model.LinearRegression()
regr.fit(X, y)
predicted_price = regr.predict([[1681479000, 220.35000610351562]]) #Predict next day closing price with today's 
print(predicted_price)