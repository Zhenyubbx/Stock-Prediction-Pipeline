from google.cloud import bigquery
from google.oauth2 import service_account

def query_table_from_bq(table_name):
    CREDS = 'is3107-project-383009-f4cdb3dfb0cb.json'
    credentials = service_account.Credentials.from_service_account_file(CREDS)
    client = bigquery.Client(credentials=credentials)
    job_config = bigquery.QueryJobConfig()

    # Set the destination table
    table = client.dataset(dataset_id='is3107-project-383009.Dataset').table(table_id=f"is3107-project-383009.Dataset.{table_name}")
    job_config.destination = table
    query = f"SELECT * FROM `is3107-project-383009.Dataset.{table_name}`"
    return client.query(query).to_dataframe()