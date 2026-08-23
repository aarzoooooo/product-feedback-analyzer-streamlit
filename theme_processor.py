from transformers import pipeline


# Load the zero-shot classification model
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)


# Possible product-feedback themes
THEMES = [
    "Product Quality",
    "Product Features",
    "Product Performance",
    "Price and Value",
    "Delivery and Shipping",
    "Customer Service",
    "Packaging",
    "Design and Usability",
    "Returns and Refunds"
]


def identify_theme(reviews):

    # Remove empty reviews
    reviews = [
        review.strip()
        for review in reviews
        if review and review.strip()
    ]

    if not reviews:
        return "General Feedback", 0.0

    # Use all available reviews instead of only one review
    combined_text = " ".join(reviews)

    # Limit text length so the model doesn't receive
    # an unnecessarily large input
    combined_text = combined_text[:4000]

    result = classifier(
        combined_text,
        candidate_labels=THEMES,
        multi_label=False
    )

    theme = result["labels"][0]
    confidence = result["scores"][0]

    return theme, confidence
