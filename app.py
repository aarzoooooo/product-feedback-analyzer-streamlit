import streamlit as st
import pandas as pd

from text_cleaner import clean_text
from tfidf_processor import create_tfidf_matrix
from kmeans_processor import create_clusters
from representative_reviews import get_representative_reviews
from sentiment_processor import analyze_sentiment
from theme_processor import identify_theme


# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="Product Feedback Analyzer",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------
# TITLE
# ---------------------------------

st.title("📊 Product Feedback Analyzer")

st.write(
    "Upload customer review data to discover sentiment, "
    "recurring feedback themes, customer pain points, "
    "and actionable product insights."
)


# ---------------------------------
# FILE UPLOAD
# ---------------------------------

uploaded_file = st.file_uploader(
    "Upload your customer review CSV",
    type=["csv"]
)


if uploaded_file is not None:

    # ---------------------------------
    # LOAD DATA
    # ---------------------------------

    df = pd.read_csv(uploaded_file)

    # ---------------------------------
    # CLEAN REVIEW TEXT
    # ---------------------------------

    if "review_body" in df.columns:
    review_column = "review_body"

elif "review_text" in df.columns:
    review_column = "review_text"

else:
    st.error(
        "CSV must contain a 'review_body' or 'review_text' column."
    )
    st.stop()

df["cleaned_review"] = (
    df[review_column]
    .fillna("")
    .apply(clean_text)
)

# ---------------------------------
# SENTIMENT ANALYSIS
# ---------------------------------

sentiments, sentiment_scores = analyze_sentiment(
    df["cleaned_review"].tolist()
)

df["sentiment"] = sentiments
df["sentiment_score"] = sentiment_scores

# ---------------------------------
# TF-IDF
# ---------------------------------

tfidf_matrix, vectorizer = create_tfidf_matrix(
    df["cleaned_review"]
)

# ---------------------------------
# K-MEANS CLUSTERING
# ---------------------------------

cluster_labels, kmeans = create_clusters(
    tfidf_matrix
    )

df["cluster"] = cluster_labels

 # ---------------------------------
 # REPRESENTATIVE REVIEWS
 # ---------------------------------

 representative_reviews = get_representative_reviews(
      tfidf_matrix,
      cluster_labels,
      kmeans,
      df["cleaned_review"]
      )

  cluster_reviews = {}

   for item in representative_reviews:

        cluster_id = item["cluster"]
        review = item["review"]

        if cluster_id not in cluster_reviews:
            cluster_reviews[cluster_id] = []

        cluster_reviews[cluster_id].append(review)

    # ---------------------------------
    # IDENTIFY THEMES
    # ---------------------------------

    cluster_themes = {}

    for cluster_id, reviews in cluster_reviews.items():

        theme, confidence = identify_theme(reviews)

        cluster_themes[cluster_id] = {
            "theme": theme
        }

    # ---------------------------------
    # SUCCESS
    # ---------------------------------

    st.success(
        "Dataset uploaded and analyzed successfully!"
    )

    # ---------------------------------
    # OVERALL FEEDBACK SUMMARY
    # ---------------------------------

    st.header("📊 Overall Feedback Summary")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Reviews",
            len(df)
        )

    with col2:

        positive_percentage = (
            (df["sentiment"] == "POSITIVE").mean() * 100
        )

        st.metric(
            "Positive Reviews",
            f"{positive_percentage:.1f}%"
        )

    with col3:

        negative_percentage = (
            (df["sentiment"] == "NEGATIVE").mean() * 100
        )

        st.metric(
            "Negative Reviews",
            f"{negative_percentage:.1f}%"
        )

    # ---------------------------------
    # SENTIMENT DISTRIBUTION
    # ---------------------------------

    st.subheader("😊 Sentiment Distribution")

    sentiment_counts = df["sentiment"].value_counts()

    st.bar_chart(
        sentiment_counts
    )

    # ---------------------------------
    # KEY CUSTOMER INSIGHTS
    # ---------------------------------

    st.header(
        "🔥 Key Customer Pain Points & Insights"
    )

    # Combine clusters with same theme

    theme_data = {}

    for cluster_id, information in cluster_themes.items():

        theme = information["theme"]

        cluster_data = df[
            df["cluster"] == cluster_id
        ]

        if theme not in theme_data:

            theme_data[theme] = {
                "reviews": 0,
                "negative": 0,
                "examples": []
            }

        theme_data[theme]["reviews"] += len(
            cluster_data
        )

        theme_data[theme]["negative"] += (
            cluster_data["sentiment"] == "NEGATIVE"
        ).sum()

        theme_data[theme]["examples"].extend(
            cluster_reviews[cluster_id]
        )

    # ---------------------------------
    # DISPLAY INSIGHTS
    # ---------------------------------

    for theme, data in theme_data.items():

        total_reviews = data["reviews"]

        negative_percentage = (
            data["negative"] / total_reviews * 100
            if total_reviews > 0
            else 0
        )

        with st.expander(
            f"🔹 {theme.title()} — {total_reviews} reviews"
        ):

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Reviews",
                    total_reviews
                )

            with col2:

                st.metric(
                    "Negative Reviews",
                    f"{negative_percentage:.1f}%"
                )

            # ---------------------------------
            # PRIORITY
            # ---------------------------------

            if negative_percentage >= 50:

                st.error(
                    "🔴 High concern: More than half of "
                    "the reviews in this theme are negative."
                )

            elif negative_percentage >= 30:

                st.warning(
                    "🟠 Attention area: A notable share "
                    "of reviews in this theme are negative."
                )

            else:

                st.success(
                    "🟢 Mostly positive feedback "
                    "in this theme."
                )

            # ---------------------------------
            # REPRESENTATIVE FEEDBACK
            # ---------------------------------

            st.write(
                "**Representative Customer Feedback:**"
            )

            for review in data["examples"][:3]:

                st.write(
                    f"• {review}"
                )

            # ---------------------------------
            # RECOMMENDED ACTION
            # ---------------------------------

            st.write(
                "**💡 Recommended Action:**"
            )

            theme_lower = theme.lower()

            if "delivery" in theme_lower:

                recommendation = (
                    "Improve delivery tracking, logistics, "
                    "and estimated delivery accuracy."
                )

            elif "quality" in theme_lower:

                recommendation = (
                    "Investigate product quality issues "
                    "and strengthen quality-control processes."
                )

            elif "performance" in theme_lower:

                recommendation = (
                    "Investigate performance issues and "
                    "improve product reliability and efficiency."
                )

            elif "price" in theme_lower:

                recommendation = (
                    "Review pricing and improve the "
                    "perceived value offered to customers."
                )

            elif "customer service" in theme_lower:

                recommendation = (
                    "Improve customer support response time "
                    "and issue-resolution processes."
                )

            elif "feature" in theme_lower:

                recommendation = (
                    "Prioritize frequently requested features "
                    "and improve existing functionality."
                )

            elif "design" in theme_lower:

                recommendation = (
                    "Improve usability and simplify "
                    "the customer experience."
                )

            elif "return" in theme_lower:

                recommendation = (
                    "Simplify the return and refund process "
                    "and improve resolution time."
                )

            elif "packaging" in theme_lower:

                recommendation = (
                    "Improve packaging quality and protection "
                    "during transportation."
                )

            else:

                recommendation = (
                    "Investigate recurring issues in this "
                    "theme and prioritize improvements "
                    "based on negative customer feedback."
                )

            st.info(
                recommendation
            )
