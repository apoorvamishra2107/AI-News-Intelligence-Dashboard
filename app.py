import streamlit as st
import pandas as pd
from core.news_fetcher import fetch_news_from_rss
from core.sentiment import analyze_sentiment, label_sentiment, detect_bias
from core.topic_model import extract_keywords
from core.insights import generate_insights
from visuals.charts import sentiment_chart, trend_chart, country_chart, topic_chart
import plotly.express as px

# Page Configuration

st.set_page_config(
    page_title="AI News Intelligence Dashboard",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for UI

st.markdown("""
<style>
body {background-color: #f7f9fc;}
h1, h2, h3 {color: #4B0082;}
.stButton>button {
    background: linear-gradient(90deg,#FF7F50,#FF4500);
    color:white;
    font-weight:bold;
    border-radius:10px;
    padding:8px 16px;
}
.stButton>button:hover {
    background: linear-gradient(90deg,#FF6347,#FF1493);
}
</style>
""", unsafe_allow_html=True)

# Sidebar Inputs

st.sidebar.title("⚙️ Dashboard Settings")

# RSS Feed Input
rss_url = st.sidebar.text_input(
    "RSS Feed URL",
    "https://feeds.bbci.co.uk/news/rss.xml"
)

article_count = st.sidebar.select_slider(
    "Number of Articles",
    options=[5, 10, 15, 20, 30, 50],
    value=15
)

st.sidebar.markdown("---")

# Supported RSS Feeds 

st.sidebar.subheader("📰 Supported RSS Feeds")
rss_feeds = [
    {"name": "CNN", "url": "http://rss.cnn.com/rss/edition.rss"},
    {"name": "BBC", "url": "https://feeds.bbci.co.uk/news/rss.xml"},
    {"name": "Google News", "url": "https://news.google.com/rss"}
]

for feed in rss_feeds:
    st.sidebar.markdown(f"- [{feed['name']}]({feed['url']})", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Filter Options

st.sidebar.title("Filter Options")
filter_sentiment = st.sidebar.multiselect(
    "Sentiment Filter",
    options=["Positive", "Neutral", "Negative"],
    default=["Positive", "Neutral", "Negative"]
)

# Main Title

st.title("📰 AI News Intelligence Dashboard")
st.markdown("Professional, interactive news analytics powered by AI and RSS feeds.")

# Run Analysis

if st.button("Run Analysis"):
    try:
        # Fetching and processing data
        df = fetch_news_from_rss(rss_url, article_count)
        df["sentiment_score"] = df["summary"].apply(analyze_sentiment)
        df["sentiment"] = df["sentiment_score"].apply(label_sentiment)
        df["bias"] = df["summary"].apply(detect_bias)

        # Applying sentiment filter
        df = df[df["sentiment"].isin(filter_sentiment)]

        if df.empty:
            st.warning("No articles match the selected sentiment filter.")
        else:
            
            # Layout: Summary Cards

            pos_count = (df["sentiment"] == "Positive").sum()
            neu_count = (df["sentiment"] == "Neutral").sum()
            neg_count = (df["sentiment"] == "Negative").sum()

            col1, col2, col3 = st.columns(3)
            col1.metric("Positive News", pos_count, "🟢")
            col2.metric("Neutral News", neu_count, "⚪")
            col3.metric("Negative News", neg_count, "🔴")

            # News Table
            
            st.subheader("📝 Latest News Articles")
            st.dataframe(df[["title", "sentiment", "bias", "published", "link"]])

            # Charts Section

            st.subheader("📊 Visual Analytics")
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.plotly_chart(sentiment_chart(df), use_container_width=True)

            with chart_col2:
                st.plotly_chart(trend_chart(df), use_container_width=True)

            st.plotly_chart(country_chart(df), use_container_width=True)

            st.plotly_chart(topic_chart(df), use_container_width=True)

            # Article Links
            
            st.subheader("🔗 Article Links")
            for _, row in df.iterrows():
                st.markdown(f"- [{row['title']}]({row['link']})")

            # Insights
            
            st.subheader("💡 Insights")
            for insight in generate_insights(df):
                st.info(insight)

    except Exception as e:
        st.error(f"Error fetching or processing RSS feed: {str(e)}")




