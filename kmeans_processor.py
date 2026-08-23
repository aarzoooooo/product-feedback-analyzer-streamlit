from sklearn.cluster import KMeans


def create_clusters(tfidf_matrix, num_clusters=5):

    kmeans = KMeans(
        n_clusters=num_clusters,
        random_state=42,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(tfidf_matrix)

    return cluster_labels, kmeans
