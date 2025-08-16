from src.db_connection import test_connection, run_query_df

ok, msg = test_connection()
print(msg)
if ok:
    df = run_query_df("SELECT * FROM sales_data LIMIT 5")
    print(df)
