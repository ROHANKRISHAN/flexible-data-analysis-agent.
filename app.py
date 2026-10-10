from pathlib import Path
import tempfile
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from analytics import load_dataset, profile_dataset, clean_dataset, summarize_numeric, grouped_analysis, time_analysis
from agent import run_agent

load_dotenv()
st.set_page_config(page_title="Flexible Data Analysis Agent", page_icon="🧠", layout="wide")
st.title("🧠 Flexible Data Analysis Agent")
st.caption("Upload a CSV with your own column names. The app profiles the schema and lets you analyse detected fields.")
default_path = Path(__file__).parent / "data" / "demo_dataset.csv"

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload any CSV file", type=["csv"])
    drop_duplicates = st.checkbox("Remove exact duplicate rows", value=False)
    st.caption("No fixed business column names are required. For best results, use a CSV with a header row.")

if uploaded is not None:
    try:
        df_raw = load_dataset(uploaded)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        st.stop()
else:
    default_path = Path(__file__).resolve().parent / "demo_dataset.csv"
    df_raw = load_dataset(str(default_path))

df, schema = clean_dataset(df_raw, drop_exact_duplicates=drop_duplicates)
numeric_cols = schema["numeric_columns"]
date_cols = schema["date_columns"]
cat_cols = schema["categorical_columns"]

st.info(f"Loaded {schema['rows']:,} rows and {schema['columns']:,} columns. Detected {len(numeric_cols)} numeric, {len(date_cols)} date, {len(cat_cols)} categorical and {len(schema['identifier_columns'])} identifier columns.")
m1,m2,m3,m4=st.columns(4)
m1.metric("Rows",f"{len(df):,}")
m2.metric("Columns",f"{len(df.columns):,}")
m3.metric("Duplicate rows",f"{int(df.duplicated().sum()):,}")
m4.metric("Columns with missing values",f"{sum(df[c].isna().any() for c in df.columns):,}")

with st.expander("Detected schema and data quality", expanded=False):
    schema_df=pd.DataFrame([{"Column":c,"Detected role":schema["roles"].get(c,"unknown"),"Data type":str(df[c].dtype),"Missing":int(df[c].isna().sum()),"Unique values":int(df[c].nunique(dropna=True))} for c in df.columns])
    st.dataframe(schema_df,use_container_width=True,hide_index=True)

tab1,tab2,tab3=st.tabs(["Ask the agent","Explore any columns","Preview data"])
with tab1:
    question=st.text_input("Ask a question about your data",placeholder="For example: compare NetSalesAmount by MarketArea")
    st.caption("Use actual column names or values where possible. With an API key, the LLM can interpret more flexible phrasing.")
    if st.button("Analyse",type="primary") and question.strip():
        result=run_agent(df,question.strip())
        st.write(result.get("summary","Analysis completed."))
        if result.get("note"): st.info(result["note"])
        if isinstance(result.get("data"),pd.DataFrame):
            st.dataframe(result["data"],use_container_width=True,hide_index=True)
        elif isinstance(result.get("data"),dict):
            st.json(result["data"])
        if result.get("plan"): st.caption("Selected analysis plan: "+str(result["plan"]))
        st.caption("Mode: "+result.get("mode","unknown"))
with tab2:
    if numeric_cols:
        metric=st.selectbox("Choose a numeric column",numeric_cols,key="metric")
        st.subheader(f"Summary statistics: {metric}")
        st.json(summarize_numeric(df,metric))
        if cat_cols:
            dim=st.selectbox("Group by categorical column",cat_cols,key="dim")
            agg=st.selectbox("Aggregation",["sum","mean","median","min","max","count"])
            if st.button("Run grouped analysis"):
                try: st.dataframe(grouped_analysis(df,dim,metric,agg),use_container_width=True,hide_index=True)
                except Exception as exc: st.error(str(exc))
        if date_cols:
            date_col=st.selectbox("Trend date column",date_cols)
            time_agg=st.selectbox("Time aggregation",["sum","mean","count"])
            if st.button("Run time analysis"):
                try:
                    trend=time_analysis(df,date_col,metric,time_agg)
                    st.dataframe(trend,use_container_width=True,hide_index=True)
                    st.line_chart(trend.set_index("Period"))
                except Exception as exc: st.error(str(exc))
    else:
        st.warning("No numeric columns were detected. You can still inspect and group categorical data, but numeric summaries need a measurable field.")
with tab3:
    st.dataframe(df.head(200),use_container_width=True,hide_index=True)
    st.caption("Showing up to the first 200 rows.")
