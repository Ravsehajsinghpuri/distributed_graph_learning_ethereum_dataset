from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd

key_path = "/home/diml_2022/tonal-nova-345908-ca95d8f2884a.json"

credentials = service_account.Credentials.from_service_account_file(
    key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
)
client = bigquery.Client(credentials=credentials)

query = """
SELECT 
  transaction_index, from_address, to_address, value
FROM
  `bigquery-public-data.crypto_ethereum.transactions`
LIMIT 100000000;
"""

query_job = client.query(query)

iterator = query_job.result(timeout=50000)
rows = list(iterator)

df = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))
df.to_csv("transaction_data_100M.csv")