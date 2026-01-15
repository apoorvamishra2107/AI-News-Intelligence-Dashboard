import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer

# Sentiment Distribution Pie Chart

def sentiment_chart(df):
    # Counting sentiment occurrences
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']  # Rename columns explicitly

    fig = px.pie(
        sentiment_counts,
        names='sentiment',
        values='count',
        color='sentiment',
        color_discrete_map={'Positive':'green','Neutral':'gray','Negative':'red'},
        title="Sentiment Distribution"
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(margin=dict(t=40,b=20))
    return fig

# Sentiment Trend Over Time Line Chart

def trend_chart(df):
    df_copy = df.copy()
    df_copy["published"] = pd.to_datetime(df_copy["published"], errors='coerce')
    df_copy = df_copy.dropna(subset=["published"])
    df_grouped = df_copy.groupby(df_copy["published"].dt.date)["sentiment_score"].mean().reset_index()

    fig = px.line(
        df_grouped,
        x='published',
        y='sentiment_score',
        markers=True,
        title="Sentiment Trend Over Time"
    )
    fig.update_layout(margin=dict(t=40,b=20))
    return fig

# Country-Wise Sentiment Chart

def country_chart(df):
    # Determine country from link
    def get_country(link):
        if "cnn.com" in link: return "USA"
        elif "bbc.co.uk" in link: return "UK"
        elif "google.com" in link: return "Global"
        return "Other"

    df["country"] = df["link"].apply(get_country)

    # Grouping by country and sentiment
    country_counts = df.groupby(['country', 'sentiment']).size().reset_index(name='count')

    fig = px.bar(
        country_counts,
        x='country',
        y='count',
        color='sentiment',
        barmode='group',
        color_discrete_map={'Positive':'green','Neutral':'gray','Negative':'red'},
        title="Country-Wise Sentiment"
    )
    fig.update_layout(margin=dict(t=40,b=20))
    return fig

# Top Keywords in News Bar Chart

def topic_chart(df):
    texts = df["summary"].tolist()
    vectorizer = TfidfVectorizer(stop_words="english", max_features=15)
    vectorizer.fit(texts)
    keywords = vectorizer.get_feature_names_out()
    counts = vectorizer.transform(texts).toarray().sum(axis=0)

    topic_df = pd.DataFrame({"Keyword": keywords, "Count": counts})

    fig = px.bar(
        topic_df,
        x='Keyword',
        y='Count',
        title="Top Keywords in News",
        text='Count',
        color='Count',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(margin=dict(t=40,b=20))
    return fig


