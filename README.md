# IS3107_Project AY2022/2023 Semester 2

Stock Prediction and Sentiment Analysis

## Information

Group: 34  
Prepared by: Chin Zhen Yu, Darryl Ee, Khoo Jing Zhi, Shannon Woo Shu En, Xu Pengcheng

## Folder/File Structure

DAGs folder:  
batch_data.py (batch data) & real_time_data.py(real time data)
NOTE: change line 14 of batch_data.py to your path directory to read csv.  
stock_tweets.csv (Batch Data from Kaggle)

Data Extraction:  
Archived python files for data extraction

stock_price_predictor:  
Contains frontend javascript files for application simulation

Machine Learning folder:  
Trains the 4 machine learning models  
Contains backend python file for application simulation

looker_studio_dashboard_link.txt:  
Link to Looker Studio Dashboard

## Steps to get airflow pipeline working

virtualenv env  
source env/bin/activate

pip install apache-airflow==2.2.3 --constraint https://raw.githubusercontent.com/apache/airflow/constraints-2.2.3/constraints-3.8.txt

pip install textblob  
pip install langdetect  
pip install matplotlib  
pip install pandas.gbq  
pip install pandas  
pip install nltk

airflow webserver --port 8081 -D

airflow scheduler

Run the real_time_data_dag and batch_data_dag tasks.

## Steps to start simulation of Frontend React Application and Backend Flask Application

Ensure that all dependencies listed in requirements.txt are installed

Frontend  
install npm on ur machine  
run "npm i" in the folder "stock_price_predictor" to install the necessary node modules  
run "npm start" to start the web application on localhost:3000

Backend  
ensure that flask is installed  
run "python3 backend.py" in the MachineLearning folder to start the backend application on localhost:5000
