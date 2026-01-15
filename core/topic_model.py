from sklearn.feature_extraction.text import TfidfVectorizer

def extract_keywords(texts, top_n=15):
    """
    Extract top keywords from a list of texts using TF-IDF.
    """
    if not texts or len(texts) == 0:
        return []

    vectorizer = TfidfVectorizer(stop_words="english", max_features=top_n)
    vectorizer.fit(texts)
    return vectorizer.get_feature_names_out()

