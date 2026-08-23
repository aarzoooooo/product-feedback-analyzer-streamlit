from theme_processor import identify_theme

reviews = [
    "The battery dies very quickly.",
    "Charging takes too long.",
    "The battery life is disappointing."
]

theme, score = identify_theme(reviews)

print("Theme:", theme)
print("Confidence:", score)
