import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text: str) -> float:
    """Return sentiment score between -1 (negative) and +1 (positive)."""
    return sia.polarity_scores(text)["compound"]

def label_sentiment(score: float) -> str:
    """Categorize sentiment score as Positive, Neutral, or Negative."""
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    return "Neutral"

def detect_bias(text: str) -> str:
    """
    Simple bias detection using TextBlob subjectivity.
    Returns 'Biased' if subjectivity > 0.5, else 'Neutral'.
    """
    blob = TextBlob(text)
    subj = blob.sentiment.subjectivity
    return "Biased" if subj > 0.5 else "Neutral"

