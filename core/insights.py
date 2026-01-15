def generate_insights(df):
    """
    Generate human-readable insights from news sentiment data.
    """
    insights = []
    if df.empty:
        return ["No data available for insights."]

    sentiment_counts = df["sentiment"].value_counts()

    if sentiment_counts.idxmax() == "Negative":
        insights.append("Overall news sentiment is negative. Potential public concern detected.")
    elif sentiment_counts.idxmax() == "Positive":
        insights.append("News sentiment is largely positive. Optimistic trends observed.")
    else:
        insights.append("News sentiment is mostly neutral.")

    insights.append(f"Most frequent sentiment: {sentiment_counts.idxmax()}")
    return insights

