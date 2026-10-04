import psycopg2
import os

psycopg2.connect(
    host=os.getenv("TIGER_DB_HOST"),
    port=os.getenv("TIGER_DB_PORT"),
    dbname=os.getenv("TIGER_DB_NAME"),
    user=os.getenv("TIGER_DB_USER"),
    password=os.getenv("TIGER_DB_PASSWORD"),  # temporary — delete this line after
    sslmode="require",
).close()
print("OK! Password is valid")
