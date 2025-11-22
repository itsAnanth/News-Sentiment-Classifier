import os
import time
import textwrap
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.ml import PipelineModel
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.sql.types import DoubleType

# override default styles
st.set_page_config(
    page_title="Real‑Time News Sentiment",
    page_icon="🗞️",
    layout="wide",
)

st.markdown(
    """
    <style>
      /* tighter default container */
      .block-container {padding-top: 1.2rem; padding-bottom: 3rem;}
      /* soft cards */
      .app-card {background: var(--background-color); border: 1px solid rgba(49,51,63,0.2); padding: 1rem 1.2rem; border-radius: 14px;}
      .muted {opacity: .7}
      .headline {font-weight: 600}
      .pill {display:inline-block; padding:.2rem .6rem; border-radius:999px; font-size:.75rem; border:1px solid rgba(49,51,63,.2)}
      .pill.pos {background: rgba(0,204,150,.12); border-color: rgba(0,204,150,.35)}
      .pill.neg {background: rgba(239,85,59,.12); border-color: rgba(239,85,59,.35)}
    </style>
    """,
    unsafe_allow_html=True,
)

# env variables
NEWSAPI_KEY_ENV = os.getenv("NEWSAPI_KEY", "")
MODEL_DIR = os.getenv("MODEL_DIR", "/app/model")
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")
SPARK_UI_PORT = os.getenv("SPARK_UI_PORT", "4040")

# sidebar
st.sidebar.header("Controls")
api_key = st.sidebar.text_input("NewsAPI Key", value=NEWSAPI_KEY_ENV, type="password", help="Read from env NEWSAPI_KEY if set.")

query = st.sidebar.text_input("Topic / Query", value="finance", help="Any NewsAPI search query.")

# Date range (default: last 24h)
end_dt = datetime.now(timezone.utc)
start_dt = end_dt - timedelta(days=1)
start_date, end_date = st.sidebar.date_input(
    "Date range (UTC)",
    value=(start_dt.date(), end_dt.date()),
)

max_articles = st.sidebar.slider("Max articles", min_value=10, max_value=100, value=40, step=10)

sort_by = st.sidebar.selectbox("Sort by", ["publishedAt", "relevancy", "popularity"], index=0)

domains_opt = [
    "bbc.co.uk", "bbc.com", "cnn.com", "reuters.com", "bloomberg.com", "ft.com",
    "wsj.com", "theguardian.com", "economist.com", "techcrunch.com", "forbes.com",
]

chosen_domains: List[str] = st.sidebar.multiselect(
    "Restrict to domains (optional)", options=domains_opt, default=[]
)

run_btn = st.sidebar.button("⚡ Fetch & Analyze", use_container_width=True)

# caching functions for speed
@st.cache_resource(show_spinner=False)
def get_spark():
    return (
        SparkSession.builder
        .appName("NewsSentimentDashboard")
        .master(SPARK_MASTER)
        .config("spark.ui.port", SPARK_UI_PORT)
        .getOrCreate()
    )

@st.cache_resource(show_spinner=False)
def load_model(model_dir: str):
    return PipelineModel.load(model_dir)

@st.cache_data(show_spinner=False, ttl=600)
def fetch_news(query: str, start: datetime, end: datetime, api_key: str, limit: int, sort_by: str, domains: Optional[List[str]]):
    if not api_key:
        return pd.DataFrame(), {"error": "Missing NewsAPI key."}

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "to": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pageSize": limit,
        "sortBy": sort_by,
        "language": "en",
        "apiKey": api_key,
    }
    if domains:
        params["domains"] = ",".join(domains)

    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        arts = data.get("articles", [])
        rows = []
        for a in arts:
            rows.append({
                "title": a.get("title"),
                "description": a.get("description"),
                "url": a.get("url"),
                "image": a.get("urlToImage"),
                "source": (a.get("source") or {}).get("name", "Unknown"),
                "publishedAt": a.get("publishedAt"),
            })
        df = pd.DataFrame(rows)
        return df, {"status": r.status_code, "total": len(df)}
    except requests.exceptions.RequestException as e:
        return pd.DataFrame(), {"error": str(e)}

# udf - user defined function, to create custom aggregate functions for usability :)
def predict_sentiment(headlines: pd.DataFrame, spark: SparkSession, model: PipelineModel) -> pd.DataFrame:
    if headlines.empty or "title" not in headlines.columns:
        return pd.DataFrame(columns=["text", "sentiment", "probability", "pos_prob", "neg_prob"]) 

    # Define UDFs here to avoid requiring an active SparkContext at import time
    extract_pos = udf(lambda v: float(v[1]) if v is not None else 0.0, DoubleType())
    extract_neg = udf(lambda v: float(v[0]) if v is not None else 0.0, DoubleType())


    spark_df = spark.createDataFrame(headlines[["title"]].rename(columns={"title": "text"}))

    # Manual TF‑IDF (ensure consistency with your training pipeline)
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    hashing = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=2000)
    idf = IDF(inputCol="rawFeatures", outputCol="features")

    words = tokenizer.transform(spark_df)
    tf = hashing.transform(words)
    idf_model = idf.fit(tf)
    feats = idf_model.transform(tf)

    # Use the last stage as classifier (e.g., LogisticRegressionModel)
    clf = model.stages[-1]
    preds = clf.transform(feats)

    preds = preds.withColumn("pos_prob", extract_pos(preds["probability"]))
    preds = preds.withColumn("neg_prob", extract_neg(preds["probability"]))

    pdf = preds.select("text", "prediction", "probability", "pos_prob", "neg_prob").toPandas()
    pdf["sentiment"] = np.where(pdf["pos_prob"] >= pdf["neg_prob"], "Positive", "Negative")
    return pdf

# page header
st.title("🗞️ Real‑Time News Sentiment Dashboard")
st.caption("Powered by PySpark ML • Live headlines from NewsAPI • Fast, clean, and responsive UI")


# load spark and model, then cache it if its the first load
if run_btn:
    with st.spinner("Starting Spark & loading model…"):
        spark = get_spark()
        try:
            model = load_model(MODEL_DIR)
        except Exception as e:
            st.error(f"Failed to load model from {MODEL_DIR}: {e}")
            st.stop()

    with st.spinner("Fetching headlines…"):
        s_dt = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        e_dt = datetime.combine(end_date, datetime.max.time()).replace(tzinfo=timezone.utc)
        news_df, meta = fetch_news(query, s_dt, e_dt, api_key, max_articles, sort_by, chosen_domains)

    if meta.get("error"):
        st.error(meta["error"])
    elif news_df.empty:
        st.warning("No articles found. Try a broader query or wider date range.")
    else:
        st.session_state["news_df"] = news_df
        st.success(f"Fetched {meta.get('total', 0)} articles • Sorted by {sort_by}")

        with st.spinner("Scoring sentiments with PySpark…"):
            preds_df = predict_sentiment(news_df, spark, model)
        if preds_df.empty:
            st.error("Prediction failed or empty.")
        else:
            st.session_state["preds_df"] = preds_df
            st.session_state["last_refreshed"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
            st.toast("Analysis complete ✅", icon="✅")

# show results
if "preds_df" in st.session_state:
    preds = st.session_state["preds_df"].copy()
    news = st.session_state.get("news_df", pd.DataFrame()).copy()

    # Merge lightweight article metadata for display
    if not news.empty:
        preds = preds.merge(news, left_on="text", right_on="title", how="left")

    # Confidence column for display
    def fmt_conf(v):
        try:
            return f"{float(v[1]) * 100:.1f}%"  # assumes [neg,pos]
        except Exception:
            return "—"
    preds["confidence"] = preds["probability"].apply(fmt_conf)

    # KPIs
    pos = int((preds["sentiment"] == "Positive").sum())
    neg = int((preds["sentiment"] == "Negative").sum())
    total = int(len(preds))

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Articles analyzed", total)
    with k2:
        st.metric("Positive", pos)
    with k3:
        st.metric("Negative", neg)
    with k4:
        st.metric("Last refreshed", st.session_state.get("last_refreshed", "—"))

    # Tabs for different views
    tab_table, tab_charts, tab_cards = st.tabs(["📋 Table", "📈 Charts", "📰 Article cards"]) 

    with tab_table:
        st.subheader("Predictions table")
        view_cols = [
            "text", "sentiment", "confidence", "source", "publishedAt", "url",
        ]
        show = preds[view_cols].rename(columns={"text": "headline"})
        st.dataframe(show, use_container_width=True, hide_index=True)

        csv = show.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, file_name="news_sentiment.csv", mime="text/csv")

    with tab_charts:
        st.subheader("Sentiment distribution")
        pie = px.pie(
            preds, names="sentiment", title="Share of headlines",
            color="sentiment", color_discrete_map={"Positive": "#00CC96", "Negative": "#EF553B"},
        )
        st.plotly_chart(pie, use_container_width=True)

        st.subheader("Counts by sentiment")
        counts = preds["sentiment"].value_counts().reset_index()
        counts.columns = ["sentiment", "count"]
        bar = px.bar(counts, x="sentiment", y="count", text="count")
        st.plotly_chart(bar, use_container_width=True)

    with tab_cards:
        st.subheader("Headlines")
        if preds.empty:
            st.info("Nothing to show.")
        else:
            for _, row in preds.iterrows():
                sent = row.get("sentiment", "?")
                pill_class = "pos" if sent == "Positive" else "neg"
                published = row.get("publishedAt") or ""
                link = row.get("url") or ""
                source = row.get("source") or "Unknown"
                desc = row.get("description") or ""

                with st.container():
                    st.markdown(
                        f"""
                        <div class='app-card'>
                          <div class='pill {pill_class}'>{sent}</div>
                          <div class='headline' style='margin-top:.4rem'>{row['text']}</div>
                          <div class='muted' style='margin:.25rem 0 .5rem 0'>{source} • {published}</div>
                          <div style='margin-bottom:.6rem'>{desc}</div>
                          <a href='{link}' target='_blank'>Open article ↗</a>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# Footer / Help
with st.expander("ℹ️ Help & Notes"):
    st.markdown(
        """
        - **API key**: Provide a NewsAPI key in the sidebar or via the `NEWSAPI_KEY` env var.
        - **Model**: The app expects a PySpark `PipelineModel` at `MODEL_DIR` env path (default `/app/model`).
        - **Probabilities**: This UI assumes the classifier's probability vector is `[neg, pos]`. If your training used a different label order, adjust the UDFs.
        - **Performance**: Fetches are cached for 10 minutes. Spark session & model are cached for reuse.
        - **Docker**: Run with: `docker run -p 8501:8501 -e NEWSAPI_KEY=... -v $PWD/model:/app/model yourimage`
        """
    )
