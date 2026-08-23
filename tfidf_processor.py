from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_matrix(reviews):
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(reviews)

    return tfidf_matrix, vectorizer
