# app/streamlit_app.py
# ---------------------
# Sales Insights Dashboard (MySQL + Streamlit + Plotly)
# Resume/Portfolio-ready single-file Streamlit app with advanced BI features

import sys
import os
from typing import Tuple, Dict, Any, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import datetime as dt
import math

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.db_connection import run_query_df  # user-provided DB helper; signature: (sql, params=None) -> pd.DataFrame

# -----------------------
# Page config & helpers
# -----------------------
st.set_page_config(page_title="Sales Insights Dashboard", page_icon="📈", layout="wide")
st.title("📊 Sales Insights Dashboard")
st.caption("Live MySQL-backed dashboard — Python, Pandas, Plotly, Streamlit")

# -----------------------
# Misc helpers
# -----------------------
@st.cache_data(ttl=300)
def available_columns() -> List[str]:
    """Return list of columns for sales_data in current database schema.
    Uses information_schema to introspect table. Works when DB user has access to information_schema.
    """
    sql = """
    SELECT COLUMN_NAME
    FROM information_schema.columns
    WHERE table_name = 'sales_data' AND table_schema = DATABASE()
    ORDER BY ORDINAL_POSITION
    """
    try:
        df = run_query_df(sql)
        return df['COLUMN_NAME'].astype(str).str.lower().tolist()
    except Exception:
        # If introspection fails, fall back to an empty list; app will still work but optional filters disabled
        return []


def col_exists(col: str) -> bool:
    return col.lower() in available_columns()


@st.cache_data(ttl=300)
def get_distinct_values(col: str, limit: int = 1000) -> List[Any]:
    """Return distinct values of column if it exists, else ['All'] only.
    Upper limit prevents huge dropdowns.
    """
    col_l = col.lower()
    if not col_exists(col_l):
        return ["All"]
    sql = f"SELECT DISTINCT {col} as val FROM sales_data WHERE {col} IS NOT NULL ORDER BY val LIMIT :lim"
    try:
        df = run_query_df(sql, params={"lim": limit})
        vals = df['val'].astype(str).tolist()
        return ["All"] + vals
    except Exception:
        return ["All"]


# -----------------------
# Sidebar filters
# -----------------------
st.sidebar.header("🔎 Filters")

# Date bounds
@st.cache_data(ttl=300)
def get_date_bounds() -> Tuple[dt.date, dt.date]:
    sql = "SELECT MIN(orderdate) AS min_d, MAX(orderdate) AS max_d FROM sales_data"
    df = run_query_df(sql)
    if df.empty or pd.isna(df.loc[0, 'min_d']) or pd.isna(df.loc[0, 'max_d']):
        return dt.date(2000, 1, 1), dt.date.today()
    return pd.to_datetime(df.loc[0, 'min_d']).date(), pd.to_datetime(df.loc[0, 'max_d']).date()

min_date, max_date = get_date_bounds()

date_range = st.sidebar.date_input("Order date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

# Year selector: if yearid exists use it else derive from orderdate
yrs = ["All"]
if col_exists('yearid'):
    yrs = get_distinct_values('yearid')
else:
    # derive from orderdate
    @st.cache_data(ttl=300)
    def get_years_from_orderdate():
        sql = "SELECT DISTINCT YEAR(orderdate) AS yr FROM sales_data WHERE orderdate IS NOT NULL ORDER BY yr"
        try:
            df = run_query_df(sql)
            return ["All"] + df['yr'].dropna().astype(int).astype(str).tolist()
        except Exception:
            return ["All"]

    yrs = get_years_from_orderdate()

year_choice = st.sidebar.selectbox("Year", options=yrs, index=0)

territory_choices = get_distinct_values('territory')
territory_choice = st.sidebar.selectbox("Territory / Region", options=territory_choices, index=0)

productline_choices = get_distinct_values('productline')
productline_choice = st.sidebar.selectbox("Product Line", options=productline_choices, index=0)

# Optional columns from schema
customer_seg_choices = get_distinct_values('customer_segment') if col_exists('customer_segment') else ["All"]
customer_segment = st.sidebar.selectbox("Customer Segment", options=customer_seg_choices, index=0) if col_exists('customer_segment') else "All"

sales_channel_choices = get_distinct_values('sales_channel') if col_exists('sales_channel') else ["All"]
sales_channel = st.sidebar.selectbox("Sales Channel", options=sales_channel_choices, index=0) if col_exists('sales_channel') else "All"

status_choices = get_distinct_values('status')
status_choice = st.sidebar.selectbox("Order Status", options=status_choices, index=0)

# Charting options
chart_lib = st.sidebar.selectbox("Chart Library", ["Plotly (recommended)", "Matplotlib (static)"], index=0)
show_map = st.sidebar.checkbox("Show geographic map (if country/territory available)", value=True)

st.sidebar.markdown("---")

# -----------------------
# Build WHERE clause
# -----------------------

def _build_where(params_in: Dict = None) -> Tuple[str, Dict]:
    clauses = []
    params: Dict[str, Any] = {} if params_in is None else params_in.copy()

    if date_range and len(date_range) == 2:
        clauses.append("orderdate BETWEEN :start_date AND :end_date")
        params['start_date'] = pd.to_datetime(date_range[0]).date()
        params['end_date'] = pd.to_datetime(date_range[1]).date()

    if year_choice and year_choice != 'All':
        # year_choice might be from yearid or from string of year
        try:
            yr = int(year_choice)
            # prefer yearid if exists
            if col_exists('yearid'):
                clauses.append("yearid = :year")
                params['year'] = yr
            else:
                clauses.append("YEAR(orderdate) = :year")
                params['year'] = yr
        except Exception:
            pass

    if territory_choice and territory_choice != 'All':
        clauses.append("territory = :territory")
        params['territory'] = territory_choice

    if productline_choice and productline_choice != 'All':
        clauses.append("productline = :productline")
        params['productline'] = productline_choice

    if customer_segment and customer_segment != 'All' and col_exists('customer_segment'):
        clauses.append("customer_segment = :custseg")
        params['custseg'] = customer_segment

    if sales_channel and sales_channel != 'All' and col_exists('sales_channel'):
        clauses.append("sales_channel = :sch")
        params['sch'] = sales_channel

    if status_choice and status_choice != 'All':
        clauses.append("status = :st")
        params['st'] = status_choice

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


# -----------------------
# KPI loaders
# -----------------------
@st.cache_data(ttl=300)
def load_kpis() -> Dict[str, Any]:
    where_sql, params = _build_where()

    # Base KPIs
    sql = f"""
    SELECT
      SUM(sales) AS total_sales,
      COUNT(DISTINCT ordernumber) AS total_orders,
      COUNT(DISTINCT customername) AS total_customers,
      SUM(quantityordered) AS total_units
    FROM sales_data
    {where_sql}
    """
    df = run_query_df(sql, params=params)
    row = df.loc[0] if not df.empty else pd.Series(dtype='float64')

    total_sales = float(row.get('total_sales', 0) or 0)
    total_orders = int(row.get('total_orders', 0) or 0)
    total_customers = int(row.get('total_customers', 0) or 0)
    total_units = int(row.get('total_units', 0) or 0)

    aov = (total_sales / total_orders) if total_orders else 0.0

    # Profit if exists
    profit = None
    profit_margin = None
    if col_exists('profit'):
        sqlp = f"SELECT SUM(profit) AS profit FROM sales_data {_build_where()[0]}"
        dfp = run_query_df(sqlp, params=_build_where()[1])
        if not dfp.empty:
            profit = float(dfp.loc[0, 'profit'] or 0)
            profit_margin = (profit / total_sales * 100) if total_sales else None

    # Return / refund rate if status indicates returns
    return_rate = None
    if col_exists('status'):
        # look for common return-like statuses
        returned_labels = ["returned", "refunded", "return", "cancelled", "canceled"]
        sqlr = f"SELECT COUNT(DISTINCT ordernumber) as returned_orders FROM sales_data {_build_where()[0]} AND (LOWER(status) IN ({', '.join([':s'+str(i) for i in range(len(returned_labels))])}))"
        # prepare params
        where_sql, params = _build_where()
        if where_sql.strip():
            # need to append AND clause; handled above
            pass
        # But easier: count matches using LIKE OR
        conds = " OR ".join([f"LOWER(status) LIKE :rl{i}" for i in range(len(returned_labels))])
        sqlr = f"SELECT COUNT(DISTINCT ordernumber) as returned_orders FROM sales_data {where_sql} "
        if conds:
            sqlr += f" AND ({conds})"
        # build params
        params2 = params.copy()
        for i,lab in enumerate(returned_labels):
            params2[f"rl{i}"] = f"%{lab}%"
        try:
            dfr = run_query_df(sqlr, params=params2)
            returned_orders = int(dfr.loc[0, 'returned_orders'] or 0) if not dfr.empty else 0
            return_rate = (returned_orders / total_orders * 100) if total_orders else 0.0
        except Exception:
            return_rate = None

    return {
        'total_sales': total_sales,
        'total_orders': total_orders,
        'total_customers': total_customers,
        'total_units': total_units,
        'aov': aov,
        'profit': profit,
        'profit_margin': profit_margin,
        'return_rate': return_rate,
    }


kpis = load_kpis()

# KPI display
k1, k2, k3, k4, k5 = st.columns([1.5,1,1,1,1])
with k1:
    st.metric(label="Total Revenue", value=f"${kpis['total_sales']:,.2f}")
with k2:
    st.metric(label="Total Orders", value=f"{kpis['total_orders']:,}")
with k3:
    st.metric(label="Total Customers", value=f"{kpis['total_customers']:,}")
with k4:
    st.metric(label="Avg Order Value (AOV)", value=f"${kpis['aov']:,.2f}")
with k5:
    if kpis['profit'] is not None:
        st.metric(label="Profit", value=f"${kpis['profit']:,.2f}", delta=f"{kpis['profit_margin']:.2f}%" if kpis['profit_margin'] is not None else None)
    else:
        st.write("\n")

if kpis['return_rate'] is not None:
    st.caption(f"Return / Refund Rate: {kpis['return_rate']:.2f}%")

st.markdown("---")

# -----------------------
# Time-series & YoY
# -----------------------
@st.cache_data(ttl=300)
def load_time_series(freq: str = 'M') -> pd.DataFrame:
    """freq: 'D' daily, 'W' weekly, 'M' monthly"""
    date_format = {
        'D': "%Y-%m-%d",
        'W': "%Y-%u",  # year-week number
        'M': "%Y-%m"
    }[freq]
    where_sql, params = _build_where()
    sql = f"SELECT DATE_FORMAT(orderdate, '{date_format}') AS period, SUM(sales) AS sales, COUNT(DISTINCT ordernumber) AS orders FROM sales_data {where_sql} GROUP BY period ORDER BY period"
    df = run_query_df(sql, params=params)
    if not df.empty:
        df['period'] = pd.to_datetime(df['period'].astype(str), errors='coerce')
    return df

# Controls for time-series
st.subheader("Revenue Trends & Comparisons")
col_ts1, col_ts2, col_ts3 = st.columns([2,1,1])
with col_ts1:
    ts_freq = st.selectbox("Time aggregation", options=['M','W','D'], format_func=lambda x: {'M':'Monthly','W':'Weekly','D':'Daily'}[x])
with col_ts2:
    show_yoy = st.checkbox("Show YoY comparison (current vs previous year)", value=True)
with col_ts3:
    smoothing = st.slider("Smoothing (rolling window months)", min_value=1, max_value=6, value=1)

df_ts = load_time_series(freq=ts_freq)

if df_ts.empty:
    st.info("No time-series data for the selected filters.")
else:
    fig = px.line(df_ts, x='period', y='sales', title='Revenue over time', markers=True)
    if smoothing and smoothing > 1:
        df_ts['sales_smooth'] = df_ts['sales'].rolling(window=smoothing, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=df_ts['period'], y=df_ts['sales_smooth'], mode='lines', name=f'Smoothed (w={smoothing})'))
    st.plotly_chart(fig, use_container_width=True)

    # YoY
    if show_yoy:
        # build queries for current year & previous year
        try:
            # derive selected year
            if year_choice != 'All':
                sel_year = int(year_choice)
            else:
                # pick max year in selection date range
                sel_year = pd.to_datetime(date_range[1]).year
            prev_year = sel_year - 1

            # fetch monthly sums for both years
            where_sql, params = _build_where()
            # replace year param
            params_prev = params.copy()
            params_curr = params.copy()
            params_prev['y'] = prev_year
            params_curr['y'] = sel_year

            # use YEAR(orderdate) = :y in where clause (append)
            base_where, base_params = _build_where()
            # supress existing year clause by removing 'yearid' if present
            # easiest: add additional condition YEAR(orderdate) = :y
            sql_y = f"SELECT DATE_FORMAT(orderdate, '%Y-%m') as ym, SUM(sales) as sales FROM sales_data {base_where} AND YEAR(orderdate) = :y GROUP BY ym ORDER BY ym"
            df_curr = run_query_df(sql_y, params={**base_params, 'y': sel_year})
            df_prev = run_query_df(sql_y, params={**base_params, 'y': prev_year})

            if not df_curr.empty and not df_prev.empty:
                df_curr['month'] = pd.to_datetime(df_curr['ym']).dt.month
                df_prev['month'] = pd.to_datetime(df_prev['ym']).dt.month
                df_merge = pd.merge(df_curr[['month','sales']], df_prev[['month','sales']], on='month', how='outer', suffixes=(f'_{sel_year}', f'_{prev_year}')).fillna(0).sort_values('month')
                df_merge['month_name'] = df_merge['month'].apply(lambda m: dt.date(1900, m, 1).strftime('%b'))
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(x=df_merge['month_name'], y=df_merge[f'sales_{prev_year}'], name=str(prev_year)))
                fig2.add_trace(go.Bar(x=df_merge['month_name'], y=df_merge[f'sales_{sel_year}'], name=str(sel_year)))
                fig2.update_layout(barmode='group', title=f'YoY monthly revenue: {prev_year} vs {sel_year}')
                st.plotly_chart(fig2, use_container_width=True)
        except Exception:
            st.info('YoY comparison not available for selected filters.')

st.markdown('---')

# -----------------------
# Product Insights
# -----------------------
st.subheader('Product Insights')
colp1, colp2 = st.columns([2,1])

@st.cache_data(ttl=300)
def top_products(by: str = 'revenue', limit: int = 10) -> pd.DataFrame:
    where_sql, params = _build_where()
    if by == 'revenue':
        sql = f"SELECT productcode, productline, SUM(sales) AS revenue, SUM(quantityordered) AS units FROM sales_data {where_sql} GROUP BY productcode, productline ORDER BY revenue DESC LIMIT :lim"
    else:
        sql = f"SELECT productcode, productline, SUM(quantityordered) AS units, SUM(sales) AS revenue FROM sales_data {where_sql} GROUP BY productcode, productline ORDER BY units DESC LIMIT :lim"
    params = {**params, 'lim': int(limit)}
    return run_query_df(sql, params=params)

with colp1:
    tp_rev = top_products(by='revenue', limit=10)
    if tp_rev.empty:
        st.info('No product data for selected filters.')
    else:
        fig = px.bar(tp_rev.head(10), x='revenue', y='productcode', orientation='h', title='Top 10 Products by Revenue')
        st.plotly_chart(fig, use_container_width=True)

with colp2:
    tp_units = top_products(by='units', limit=10)
    if tp_units.empty:
        st.info('No product unit data for selected filters.')
    else:
        fig = px.bar(tp_units.head(10), x='units', y='productcode', orientation='h', title='Top 10 Products by Units Sold')
        st.plotly_chart(fig, use_container_width=True)

# Product line breakdown
pl_df = None
@st.cache_data(ttl=300)
def productline_breakdown():
    where_sql, params = _build_where()
    sql = f"SELECT productline, SUM(sales) AS sales, SUM(quantityordered) AS units FROM sales_data {where_sql} GROUP BY productline ORDER BY sales DESC"
    return run_query_df(sql, params=params)

pl_df = productline_breakdown()
if not pl_df.empty:
    fig = px.pie(pl_df, values='sales', names='productline', title='Product Line Sales Share')
    st.plotly_chart(fig, use_container_width=True)

st.markdown('---')

# -----------------------
# Customer Insights
# -----------------------
st.subheader('Customer Insights')

@st.cache_data(ttl=300)
def top_customers(limit: int = 10) -> pd.DataFrame:
    where_sql, params = _build_where()
    sql = f"SELECT customername, COUNT(DISTINCT ordernumber) AS orders, SUM(sales) AS revenue FROM sales_data {where_sql} GROUP BY customername ORDER BY revenue DESC LIMIT :lim"
    return run_query_df(sql, params={**params, 'lim': int(limit)})

tc = top_customers(limit=15)
if not tc.empty:
    st.write('Top customers by revenue')
    st.dataframe(tc, use_container_width=True)

# Retention and repeat orders
@st.cache_data(ttl=300)
def customer_retention():
    where_sql, params = _build_where()
    # count customers with >1 distinct orders
    sql = f"SELECT customername, COUNT(DISTINCT ordernumber) AS orders FROM sales_data {where_sql} GROUP BY customername"
    df = run_query_df(sql, params=params)
    if df.empty:
        return None
    total_customers = df.shape[0]
    repeat_customers = df[df['orders'] > 1].shape[0]
    retention_rate = (repeat_customers / total_customers * 100) if total_customers else 0
    return {
        'total_customers': total_customers,
        'repeat_customers': repeat_customers,
        'retention_rate': retention_rate
    }

ret = customer_retention()
if ret:
    c1, c2, c3 = st.columns(3)
    c1.metric('Total customers', ret['total_customers'])
    c2.metric('Repeat customers', ret['repeat_customers'])
    c3.metric('Retention (repeat rate)', f"{ret['retention_rate']:.2f}%")

st.markdown('---')

# -----------------------
# Territory & Map
# -----------------------
st.subheader('Territory & Regional Insights')

@st.cache_data(ttl=300)
def sales_by_region(limit: int = 100):
    where_sql, params = _build_where()
    if col_exists('country'):
        sql = f"SELECT country, SUM(sales) AS sales FROM sales_data {where_sql} GROUP BY country ORDER BY sales DESC LIMIT :lim"
    elif col_exists('territory'):
        sql = f"SELECT territory, SUM(sales) AS sales FROM sales_data {where_sql} GROUP BY territory ORDER BY sales DESC LIMIT :lim"
    else:
        return pd.DataFrame()
    return run_query_df(sql, params={**params, 'lim': int(limit)})

sbr = sales_by_region(limit=200)
if sbr.empty:
    st.info('No regional data available for map/chart.')
else:
    if col_exists('country') and show_map:
        # Choropleth by country (requires recognizable country names)
        fig = px.choropleth(sbr, locations='country', locationmode='country names', color='sales', hover_name='country', title='Sales by Country')
        st.plotly_chart(fig, use_container_width=True)
    else:
        # bar chart by territory
        col1, col2 = st.columns([2,1])
        with col1:
            if 'territory' in sbr.columns:
                fig = px.bar(sbr, x=sbr.columns[0], y='sales', title='Sales by Territory')
                st.plotly_chart(fig, use_container_width=True)

    # growth trends per region (top 5)
    top_regions = sbr.head(5)[sbr.columns[0]].tolist()
    if top_regions:
        # fetch monthly trends for each
        where_sql_base, params_base = _build_where()
        # region filter in loop
        df_list = []
        for r in top_regions:
            colname = 'country' if col_exists('country') else 'territory'
            sql = f"SELECT DATE_FORMAT(orderdate, '%Y-%m') as ym, SUM(sales) as sales FROM sales_data {where_sql_base} AND {colname} = :r GROUP BY ym ORDER BY ym"
            try:
                dft = run_query_df(sql, params={**params_base, 'r': r})
                if not dft.empty:
                    dft['region'] = r
                    dft['ym'] = pd.to_datetime(dft['ym'])
                    df_list.append(dft)
            except Exception:
                continue
        if df_list:
            df_growth = pd.concat(df_list)
            fig = px.line(df_growth, x='ym', y='sales', color='region', title='Growth trends for top regions')
            st.plotly_chart(fig, use_container_width=True)

st.markdown('---')

# -----------------------
# Order Insights
# -----------------------
st.subheader('Order Insights')

@st.cache_data(ttl=300)
def order_insights():
    where_sql, params = _build_where()
    sql = f"SELECT ordernumber, orderdate, quantityordered, sales, customername FROM sales_data {where_sql}"
    return run_query_df(sql, params=params)

oi = order_insights()
if not oi.empty:
    # AOV over time already shown; show orders per day-of-week
    oi['orderdate'] = pd.to_datetime(oi['orderdate'])
    oi['dow'] = oi['orderdate'].dt.day_name()
    dow = oi.groupby('dow').agg({'ordernumber':'nunique','sales':'sum'}).reset_index().rename(columns={'ordernumber':'orders'})
    # reorder days
    days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    dow['dow'] = pd.Categorical(dow['dow'], categories=days, ordered=True)
    dow = dow.sort_values('dow')
    fig = px.bar(dow, x='dow', y='orders', title='Orders per day of week')
    st.plotly_chart(fig, use_container_width=True)

st.markdown('---')

# -----------------------
# Detail Table & Exports
# -----------------------
st.subheader('Detailed Table (filtered)')

@st.cache_data(ttl=300)
def load_detail(limit: int = 2000) -> pd.DataFrame:
    where_sql, params = _build_where()
    sql = f"SELECT orderdate, ordernumber, customername, productline, productcode, quantityordered, priceeach, sales, territory, country, state, status"
    if col_exists('customer_segment'):
        sql += ", customer_segment"
    if col_exists('sales_channel'):
        sql += ", sales_channel"
    sql += f" FROM sales_data {where_sql} ORDER BY orderdate DESC LIMIT :lim"
    params = {**params, 'lim': int(limit)}
    return run_query_df(sql, params=params)

detail_df = load_detail(limit=5000)
if detail_df.empty:
    st.info('No detail rows for selected filters.')
else:
    st.dataframe(detail_df, use_container_width=True, height=500)

    # Export buttons
    csv = detail_df.to_csv(index=False).encode('utf-8')
    excel_bytes = None
    try:
        from io import BytesIO
        output = BytesIO()
        detail_df.to_excel(output, index=False, engine='openpyxl')
        excel_bytes = output.getvalue()
    except Exception:
        excel_bytes = None

    st.download_button('⬇️ Download CSV', csv, file_name='sales_filtered.csv', mime='text/csv')
    if excel_bytes:
        st.download_button('⬇️ Download Excel', excel_bytes, file_name='sales_filtered.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

st.markdown('---')

# -----------------------
# End notes & resume bullet helper
# -----------------------
st.sidebar.markdown('---')
if st.sidebar.button('Show recommended resume bullets'):
    st.sidebar.markdown('**Sales Insights Dashboard — Suggested resume bullets**')
    st.sidebar.markdown('- Designed and deployed an interactive Streamlit dashboard connected to a live MySQL database to visualize sales performance.')
    st.sidebar.markdown('- Implemented dynamic filters (date, year, territory, product line, customer segment, sales channel) and KPIs (revenue, orders, customers, AOV, profit, return rate).')
    st.sidebar.markdown('- Built interactive Plotly visualizations: time-series revenue trends, YoY comparisons, top products, customer retention metrics, and choropleth/region maps.')
    st.sidebar.markdown('- Enabled export of filtered results (CSV, Excel) and provided drill-downs for product & regional analysis.')

st.caption('© Your Company — Sales Insights Dashboard. Built with Python, Pandas, Plotly, Streamlit.')
