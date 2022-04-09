from google.cloud import bigquery
import pandas as pd

client = bigquery.Client()

query = """
SELECT 
  transaction_index, from_address, to_address, value
FROM
  `bigquery-public-data.crypto_ethereum.transactions`
LIMIT 1000000;
"""

query_job = client.query(query)

iterator = query_job.result(timeout=5000)
rows = list(iterator)

df = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))
df.to_csv("transaction_data.csv")