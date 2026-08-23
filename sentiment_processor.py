from transformers import pipeline


# Load pretrained sentiment analysis model
sentiment_model = pipeline(
    "sentiment-analysis"
)


def analyze_sentiment(reviews):
    results = sentiment_model(
        reviews,
        truncation=True,
        max_length=512
    )
    sentiments = [result["label"] for result in results]
    scores = [result["score"] for result in results]

    return sentiments, scores
