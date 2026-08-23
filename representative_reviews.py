import numpy as np


def get_representative_reviews(
    tfidf_matrix,
    cluster_labels,
    kmeans,
    reviews,
    reviews_per_cluster=3
):
    representative_reviews = []

    for cluster_id in range(kmeans.n_clusters):

        cluster_indices = np.where(
            cluster_labels == cluster_id
        )[0]

        if len(cluster_indices) == 0:
            continue

        cluster_vectors = tfidf_matrix[cluster_indices]

        centroid = kmeans.cluster_centers_[cluster_id]

        distances = np.linalg.norm(
            cluster_vectors.toarray() - centroid,
            axis=1
        )

        closest_indices = cluster_indices[
            np.argsort(distances)[:reviews_per_cluster]
        ]

        for index in closest_indices:
            representative_reviews.append({
                "cluster": cluster_id,
                "review": reviews.iloc[index]
            })

    return representative_reviews
