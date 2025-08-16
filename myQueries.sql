
CREATE DATABASE IF NOT EXISTS sales_insights;

USE sales_insights;


CREATE TABLE sales_data (
    ordernumber INT,
    quantityordered INT,
    priceeach DECIMAL(10,2),
    orderlinenumber INT,
    sales DECIMAL(15,2),
    orderdate DATE,
    status VARCHAR(10),
    quarterid INT,
    monthid INT,
    yearid INT,
    productline VARCHAR(100),
    msrp INT,
    productcode VARCHAR(50),
    customername VARCHAR(100),
    city VARCHAR(150),
    state VARCHAR(150),
    country VARCHAR(150),
    territory VARCHAR(50),
    deal_size VARCHAR(20),
    PRIMARY KEY (ordernumber, productcode)
);

SELECT * FROM sales_data;


-- Inspect a sample
SELECT orderdate FROM sales_data LIMIT 5;

SET SQL_SAFE_UPDATES = 0;

-- If orderdate is text like '31-12-2003', convert and change type:
UPDATE sales_data
SET orderdate = STR_TO_DATE(orderdate, '%d-%m-%Y')
WHERE orderdate REGEXP '^[0-9]{2}-[0-9]{2}-[0-9]{4}$';

ALTER TABLE sales_data
MODIFY COLUMN orderdate DATE;


-- Enforce numeric types (safety)
ALTER TABLE sales_data
MODIFY COLUMN quantityordered INT,
MODIFY COLUMN priceeach DECIMAL(10,2),
MODIFY COLUMN sales DECIMAL(15,2);


-- Indexes for speed
-- Goal: make filters & group-bys snappy in Streamlit.

-- Time filters & trends
CREATE INDEX idx_csd_orderdate ON sales_data(orderdate);

-- Category analysis
CREATE INDEX idx_csd_productline ON sales_data(productline);

-- Geography filters
CREATE INDEX idx_csd_territory ON sales_data(territory);
CREATE INDEX idx_csd_country_state ON sales_data(country, state);

-- Customer views (use your real customer id/name column)
CREATE INDEX idx_csd_customername ON sales_data(customername);

-- Rule of thumb: index columns you use in WHERE, JOIN, and frequent GROUP BY.



-- Create analytics views (your dashboard’s data API)
-- Goal: precompute the most common aggregations so the app stays simple & fast.
-- These views are virtual tables you can query directly.

-- 1) Monthly revenue

CREATE OR REPLACE VIEW v_monthly_revenue AS
SELECT
  DATE_FORMAT(orderdate, '%Y-%m') AS ym,
  YEAR(orderdate) AS year,
  MONTH(orderdate) AS month,
  SUM(sales) AS revenue,
  COUNT(DISTINCT ordernumber) AS orders,
  COUNT(DISTINCT customername) AS customers
FROM sales_data
GROUP BY 1,2,3;


-- 2) Revenue by product line

CREATE OR REPLACE VIEW v_revenue_by_productline AS
SELECT
  productline,
  SUM(sales) AS revenue,
  SUM(quantityordered) AS units,
  COUNT(DISTINCT ordernumber) AS orders
FROM sales_data
GROUP BY productline;


-- 3) Top customers (overall & by year)

CREATE OR REPLACE VIEW v_top_customers AS
SELECT
  customername,
  SUM(sales) AS revenue,
  COUNT(DISTINCT ordernumber) AS orders
FROM sales_data
GROUP BY customername;

CREATE OR REPLACE VIEW v_top_customers_year AS
SELECT
  YEAR(orderdate) AS year,
  customername,
  SUM(sales) AS revenue,
  COUNT(DISTINCT ordernumber) AS orders
FROM sales_data
GROUP BY 1,2;


-- 4) Geography (territory / country / state)

CREATE OR REPLACE VIEW v_revenue_by_geo AS
SELECT
  territory,
  country,
  state,
  SUM(sales) AS revenue,
  COUNT(DISTINCT ordernumber) AS orders
FROM sales_data
GROUP BY territory, country, state;
