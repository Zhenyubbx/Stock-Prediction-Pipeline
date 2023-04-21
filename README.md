# IS3107_Project AY2022/2023 Semester 2

Stock Prediction and Sentiment Analysis

## Information

Group: 34
Prepared by: Chin Zhen Yu, Darryl Ee, Khoo Jing Zhi, Shannon Woo Shu En, Xu Pengcheng

## Folder/File Structure

Airflow File: batch_data.py (batch data) & real_time_data.py(real time data) note: change line 14 of batch_data.py to your path directory to read csv.
Machine Learning:
Data Visualisation:
Google Lookers Link: google_lookers_dashboard_link.txt

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
