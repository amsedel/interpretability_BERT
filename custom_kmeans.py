
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

class KMeansPP:
    def __init__(self, n_clusters, max_iterations=300, n_init=10, random_state=42, metric_type='euclidean'):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.n_init = n_init
        self.random_state = random_state
        self.metric_type = metric_type

    def fit(self, X):
        best_kmeans = None
        best_inertia = float("inf")

        for i in range(self.n_init):
            kmeans = self.fit_once(X, i)

            if kmeans.inertia_ < best_inertia:
                best_kmeans = kmeans
                best_inertia = kmeans.inertia_

        self.centroids = best_kmeans.cluster_centers_
        self.labels = best_kmeans.labels_
        return kmeans

    def fit_once(self, X, i):
        # Inicializa los centroides utilizando K-Means++
        centroids = self.initialize_centroids(X, i)

        for _ in range(self.max_iterations):
            # Asigna puntos de datos a los clústeres
            clusters = self.assign_clusters(X, centroids)

            # Calcula nuevos centroides a partir de los puntos asignados
            new_centroids = self.calculate_centroids(X, clusters)

            # Comprueba si los centroides han convergido
            if np.all(centroids == new_centroids):
                break

            centroids = new_centroids

        return KMeansResult(centroids, clusters, np.sum((X - centroids[clusters]) ** 2))

    def initialize_centroids(self, X, i):
        n_samples, _ = X.shape
        centroids = np.empty((self.n_clusters, X.shape[1]))

        # Establece la semilla para controlar la aleatoriedad
        np.random.seed(self.random_state + i)

        # Selecciona el primer centroide aleatoriamente
        initial_centroid = X[np.random.choice(n_samples)]
        centroids[0] = initial_centroid

        for i in range(1, self.n_clusters):
            # Calcula las distancias al cuadrado desde los puntos de datos al centroide más cercano
            if self.metric_type == 'cosine':
                cosine_sims = cosine_similarity(X, centroids[:i])
                distances = 1 - np.max(cosine_sims, axis=1)
            else:
                distances = np.array([min(np.linalg.norm(x - c) for c in centroids[:i]) ** 2 for x in X])
            distances[distances < 1e-10] = 0
            # Calcula las probabilidades de selección
            probabilities = distances / np.sum(distances)
            # Selecciona el próximo centroide basado en las probabilidades
            next_centroid = X[np.random.choice(n_samples, p=probabilities)]
            centroids[i] = next_centroid

        return centroids

    def assign_clusters(self, X, centroids):
        if self.metric_type == 'cosine':
            cosine_sims = cosine_similarity(X, centroids)
            distances = cosine_sims
            return np.argmax(cosine_sims, axis=1)
        else:
            distances = np.array([np.linalg.norm(X - c, axis=1) for c in centroids])
            return np.argmin(distances, axis=0)

    def calculate_centroids(self, X, clusters):
        new_centroids = np.array([np.mean(X[clusters == i], axis=0) for i in range(self.n_clusters)])
        return new_centroids

class KMeansResult:
    def __init__(self, centroids, labels, inertia):
        self.cluster_centers_ = centroids
        self.labels_ = labels
        self.inertia_ = inertia