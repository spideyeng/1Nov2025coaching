# streamlit_app.py
import re
from datetime import datetime
from functools import lru_cache

import pandas as pd
import plotly.express as px
import streamlit as st

# -------------------------
# Helper parsers / metrics
# -------------------------
def extract_categories(text):
    """Return list of category names from the categories cell using regex."""
    if pd.isna(text) or text == "":
        return []
    # find all "category":"..."; robust to the CSV's quoting style
    return re.findall(r'"category":"([^"]+)"', text)

def expand_categories(df, categories_col='categories'):
    """Expand rows so each job-category is its own row."""
    df = df.copy()
    df['__categories'] = df[categories_col].apply(extract_categories)
    df = df.explode('__categories').rename(columns={'__categories': 'category'})
    df['category'] = df['category'].fillna('Unspecified')
    return df

def to_datetime_series(df, date_col='metadata_newPostingDate'):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=False, errors='coerce')
    return df

# -------------------------
# Load and prepare data
# -------------------------
@st.cache_data
def load_prepare(path):
    # read, handle BOM if present
    df = pd.read_csv(path, encoding='utf-8-sig')
    # parse date column and expand categories
    df = to_datetime_series(df, 'metadata_newPostingDate')
    df = expand_categories(df, 'categories')
    # ensure numeric columns
    for c in ['numberOfVacancies', 'metadata_totalNumberJobApplication', 'metadata_totalNumberOfView',
              'salary_minimum', 'salary_maximum']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    return df

# -------------------------
# Aggregation helpers
# -------------------------
def aggregate_time_series(df, metric='job_count', freq='M', date_col='metadata_newPostingDate'):
    # metric options: job_count, vacancies_sum, views_sum, applications_sum, avg_salary_min, avg_salary_max
    df = df.copy().dropna(subset=[date_col])
    df.set_index(date_col, inplace=True)
    if metric == 'job_count':
        agg = df.groupby([pd.Grouper(freq=freq), 'category'])['metadata_jobPostId'].nunique().unstack(fill_value=0)
    elif metric == 'vacancies_sum':
        agg = df.groupby([pd.Grouper(freq=freq), 'category'])['numberOfVacancies'].sum().unstack(fill_value=0)
    elif metric == 'views_sum':
        agg = df.groupby([pd.Grouper(freq=freq), 'category'])['metadata_totalNumberOfView'].sum().unstack(fill_value=0)
    elif metric == 'applications_sum':
        agg = df.groupby([pd.Grouper(freq=freq), 'category'])['metadata_totalNumberJobApplication'].sum().unstack(fill_value=0)
    elif metric == 'avg_salary_min':
        agg = df.groupby([pd.Grouper(freq=freq), 'category'])['salary_minimum'].mean().unstack(fill_value=0)
    elif metric == 'avg_salary_max':
        agg = df.groupby([pd.Grouper(freq=freq), 'category'])['salary_maximum'].mean().unstack(fill_value=0)
    else:
        raise ValueError("Unknown metric")
    agg.index = agg.index.to_timestamp() if hasattr(agg.index, "to_timestamp") else agg.index
    return agg.sort_index()

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(layout='wide', page_title='Category Trends Dashboard')

st.title('Interactive Category Trends Dashboard')

# file input
uploaded = st.file_uploader("Upload CSV (or use attached data file)", type=['csv'])
if not uploaded:
    st.info("Please upload the CSV file (use the provided SGJobData_small.csv).")
    st.stop()

# load
df = load_prepare(uploaded)

# controls
st.sidebar.header("Filters & Settings")
min_date = df['metadata_newPostingDate'].min()
max_date = df['metadata_newPostingDate'].max()
date_range = st.sidebar.date_input("Posting date range", [min_date.date(), max_date.date()])

all_categories = sorted(df['category'].unique())
selected_cats = st.sidebar.multiselect("Categories (leave empty = all)", all_categories, default=all_categories[:6])

freq_choice = st.sidebar.selectbox("Aggregation frequency", options=['D', 'W', 'M', 'Q'], index=2,
                                   format_func=lambda x: {'D':'Daily','W':'Weekly','M':'Monthly','Q':'Quarterly'}[x])
metric = st.sidebar.selectbox("Metric", options=[
    ('job_count','Job posts (unique)'),
    ('vacancies_sum','Sum of vacancies'),
    ('views_sum','Sum of views'),
    ('applications_sum','Sum of applications'),
    ('avg_salary_min','Average salary minimum'),
    ('avg_salary_max','Average salary maximum')],
    index=0, format_func=lambda t: t[1])[0]

top_n = st.sidebar.slider("Top N categories (for bar chart)", 3, 20, 8)

# filter df to date range and categories
start_dt = pd.to_datetime(date_range[0])
end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
df_filt = df[(df['metadata_newPostingDate'] >= start_dt) & (df['metadata_newPostingDate'] <= end_dt)]
if selected_cats:
    df_filt = df_filt[df_filt['category'].isin(selected_cats)]

if df_filt.empty:
    st.warning("No rows after applying filters. Adjust date range or categories.")
    st.stop()

# aggregate
agg = aggregate_time_series(df_filt, metric=metric, freq=freq_choice)

# limit categories (top N by total)
totals = agg.sum().sort_values(ascending=False)
top_categories = totals.head(top_n).index.tolist()
agg_top = agg[top_categories]

# layout plots
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Time series (interactive)")
    df_long = agg_top.reset_index().melt(id_vars=agg_top.index.name or 'index', var_name='category', value_name='value')
    # rename time column consistently
    time_col = df_long.columns[0]
    fig_ts = px.line(df_long, x=time_col, y='value', color='category', title='Trend over time', markers=True)
    fig_ts.update_layout(legend_title_text='Category', hovermode='x unified')
    st.plotly_chart(fig_ts, use_container_width=True)

    st.subheader("Stacked area (contribution over time)")
    fig_area = px.area(df_long, x=time_col, y='value', color='category', title='Stacked area - contribution', groupnorm=None)
    st.plotly_chart(fig_area, use_container_width=True)

with col2:
    st.subheader(f"Top {top_n} categories (total over period)")
    bar_df = totals.head(top_n).reset_index()
    bar_df.columns = ['category', 'total']
    fig_bar = px.bar(bar_df, x='total', y='category', orientation='h', title='Top categories', labels={'total':'Total', 'category':'Category'})
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Quick metrics")
    st.metric("Total posts", int(df_filt['metadata_jobPostId'].nunique()))
    st.metric("Total vacancies", int(df_filt['numberOfVacancies'].sum()))
    st.metric("Total views", int(df_filt['metadata_totalNumberOfView'].sum()))

st.subheader("Data table (first 200 rows)")
st.dataframe(df_filt.reset_index(drop=True).loc[:, ['metadata_jobPostId','metadata_newPostingDate','category','title','postedCompany_name','numberOfVacancies','salary_minimum','salary_maximum']].head(200))

st.caption("Notes: categories are extracted from the categories string; salary fields may be 0 if missing.")