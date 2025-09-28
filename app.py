import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, udf
from pyspark.ml import PipelineModel
import plotly.express as px
import requests
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.linalg import DenseVector

# Initialize Spark
spark = SparkSession.builder.appName("Dashboard").getOrCreate()
model = PipelineModel.load("/content/news_sentiment_model")

# Fetch function with debugging
def fetch_news_headlines(query='finance', num_articles=20):
    API_KEY = 'dc4e05a01f6b41469d7feb0ab22e65ce'  # Replace with your NewsAPI key
    url = 'https://newsapi.org/v2/everything'
    params = {'q': query, 'apiKey': API_KEY, 'sortBy': 'publishedAt', 'pageSize': num_articles}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        st.write("API Response Status:", response.status_code)
        if 'articles' not in data or not data['articles']:
            st.error("No articles found. Try broader query like 'news'.")
            return pd.DataFrame()
        headlines = []
        for article in data['articles']:
            if article.get('title'):
                headlines.append({
                    'title': article['title'],
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'publishedAt': article.get('publishedAt', '')
                })
        if not headlines:
            st.error("No valid titles extracted.")
            return pd.DataFrame()
        df = pd.DataFrame(headlines)
        st.write("Fetched Headlines:", df.head())  # Debug: Show fetched data
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Failed: {e}")
        return pd.DataFrame()
    except ValueError as e:
        st.error(f"JSON Decode Error: {e}")
        return pd.DataFrame()

# UDF to extract positive and negative probabilities
extract_pos_prob = udf(lambda v: float(v[1]) if v is not None else 0.0, "double")
extract_neg_prob = udf(lambda v: float(v[0]) if v is not None else 0.0, "double")

# Predict function (using both probabilities)
def predict_sentiment(headlines_df):
    if headlines_df.empty or 'title' not in headlines_df.columns:
        return pd.DataFrame(columns=['text', 'sentiment', 'probability'])
    # Create Spark DataFrame
    spark_df = spark.createDataFrame(headlines_df['title'], "string").withColumnRenamed("value", "text")
    # Manually apply feature extraction
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=2000)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    # Transform without fit for Tokenizer and HashingTF
    words_data = tokenizer.transform(spark_df)
    tf_data = hashingTF.transform(words_data)
    idf_model = idf.fit(tf_data)
    featured_data = idf_model.transform(tf_data)
    # Extract the classifier and predict
    classifier = model.stages[-1]  # LogisticRegressionModel
    preds = classifier.transform(featured_data)
    # Debug: Show unique prediction values
    # st.write("Prediction Values:", preds.select("prediction").distinct().collect())
    # Add extracted probabilities as columns
    preds = preds.withColumn("pos_prob", extract_pos_prob(preds["probability"]))
    preds = preds.withColumn("neg_prob", extract_neg_prob(preds["probability"]))
    preds_pd = preds.select("text", "prediction", "probability", "pos_prob", "neg_prob").toPandas()
    # Debug: Show sample probabilities
    st.write("Sample Probabilities:", preds_pd[['text', 'pos_prob', 'neg_prob']].head())
    # Adjust thresholds:
    preds_pd['sentiment'] = preds_pd.apply(
        lambda row: "Positive" if row['pos_prob'] >= row['neg_prob'] else "Negative",
        axis=1
    )
    result = preds_pd[['text', 'sentiment', 'probability']]
    if result.empty:
        st.warning("No predictions generated—check model compatibility.")
    return result

# Streamlit app
st.title("🚀 Real-Time News Sentiment Dashboard")
st.markdown("Powered by PySpark ML – Analyzing Live News!")

query = st.sidebar.text_input("News Topic", "finance")
if st.sidebar.button("Fetch & Predict"):
    with st.spinner("Analyzing sentiments..."):
        news_df = fetch_news_headlines(query)
        if not news_df.empty:
            preds_df = predict_sentiment(news_df)
            st.session_state.preds = preds_df
        else:
            st.error("No news fetched. Check API key or query.")

if 'preds' in st.session_state:
    df = st.session_state.preds
    st.subheader("Latest News Headlines")
    df['confidence'] = df['probability'].apply(lambda x: f"{x[1]*100:.1f}%")
    st.dataframe(df[['text', 'sentiment', 'confidence']], use_container_width=True)

    st.subheader("Sentiment Distribution")
    fig = px.pie(df, names='sentiment', title="Sentiment Distribution",
                 color='sentiment', color_discrete_map={'Positive': '#00CC96', 'Negative': '#EF553B', 'Neutral': '#AB63FA'})
    st.plotly_chart(fig)

    st.subheader("Sentiment Bar Chart")
    st.bar_chart(df['sentiment'].value_counts())

st.markdown("---")
st.info(f"Data from NewsAPI.org ")
