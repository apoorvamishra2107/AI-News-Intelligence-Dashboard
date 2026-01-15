import feedparser
import pandas as pd

def fetch_news_from_rss(rss_url: str, limit: int = 20) -> pd.DataFrame:
    """
    Fetch latest articles from RSS feed and return as a DataFrame.
    """
    feed = feedparser.parse(rss_url)
    if not feed.entries:
        raise ValueError("Invalid or empty RSS feed URL")

    data = []
    for entry in feed.entries[:limit]:
        data.append({
            "title": entry.get("title", "").strip(),
            "summary": entry.get("summary", "").strip(),
            "link": entry.get("link", "").strip(),
            "published": entry.get("published", "").strip()
        })

    df = pd.DataFrame(data)

    # Published column is datetime
    df["published"] = pd.to_datetime(df["published"], errors='coerce')
    df = df.dropna(subset=["published"])

    return df

