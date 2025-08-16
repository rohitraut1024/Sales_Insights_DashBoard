import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import pandas as pd
import urllib.parse

# Load environment variables
load_dotenv()

# Read from .env
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "yourpassword")  # keep raw password
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "sales_insights")

# URL-encode password safely (handles @, #, %, etc.)
MYSQL_PASSWORD = urllib.parse.quote_plus(MYSQL_PASSWORD)

# Create SQLAlchemy engine with connection pooling

engine = create_engine(
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}",
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800
)

# Function to run SELECT queries and return DataFrame
def run_query_df(sql: str, params: dict | None = None) -> pd.DataFrame:
    """
    Execute a SQL SELECT query and return results as pandas DataFrame.
    """
    with engine.connect() as conn:
        result = pd.read_sql(text(sql), conn, params=params or {})
    return result

# Function to run INSERT/UPDATE/DELETE queries
def run_query(sql: str, params: dict | None = None) -> None:
    """
    Execute a SQL command that modifies data (INSERT, UPDATE, DELETE).
    """
    with engine.begin() as conn:  # begin() handles commit/rollback automatically
        conn.execute(text(sql), params or {})

# --- TESTING ---
if __name__ == "__main__":
    try:
        df = run_query_df("SELECT * FROM sales_data LIMIT 5;")
        print(df)
    except Exception as e:
        print("❌ Database connection/test query failed:", e)
