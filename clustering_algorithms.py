from sklearn.cluster import KMeans, DBSCAN, MeanShift, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from custom_kmeans import KMeansPP
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, rand_score, mutual_info_score, fowlkes_mallows_score, homogeneity_completeness_v_measure
from tkinter import messagebox
import torch
import os
import numpy as np


class ClusteringAlgorithms:
    def __init__(self, analysis_type, method):
        if analysis_type == "Attention" and method != '1':
            self.data = torch.load(os.path.dirname(os.path.abspath(__file__))+ '/data_all_sequences_per_layer_attention.pth')
        elif analysis_type == "CLS" and method != '1':
            self.data = torch.load(os.path.dirname(os.path.abspath(__file__))+ '/data_all_sequences_per_layer_CLS.pth')
        elif analysis_type == "Attention" and method == '1':
            self.data = torch.load(os.path.dirname(os.path.abspath(__file__))+ '/data_per_layer_PCA_attention.pth')
        elif analysis_type == "CLS" and method == '1':
            self.data = torch.load(os.path.dirname(os.path.abspath(__file__))+ '/data_per_layer_PCA_CLS.pth')
        self.vectors_per_layer, self.sequences, self.labels, self.dimensions = self.data['vectors'], self.data['sequences'], self.data['labels'], self.data['dimensions']
        self.clusters_obj = []
        self._labels = []
        self.scaled = 'no-scale'

    def z_score(self, layer):
        # Crea una instancia del StandardScaler
        scaler = StandardScaler()
        # Ajusta el escalador a tus datos y transforma los datos
        normalized_data = scaler.fit_transform(self.vectors_per_layer[layer])
        return normalized_data


    def k_means_algorithm(self, k, layer, scaled='no-scale', metric='euclidean', random_state=0, n_init=30):
        if scaled == 'z-score':
            self.scaled = 'z-score'
            vectors = self.z_score(layer)
            self.vectors_z_score = vectors
        else:
            self.scaled = 'no-scale'
            vectors = self.vectors_per_layer[layer]
        if metric == 'cosine':
            kmeans = KMeansPP(n_clusters=k, random_state=random_state, n_init=n_init, metric_type=metric)
        else:
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)

        self.clusters_obj = kmeans.fit(vectors)
        self._labels = self.clusters_obj.labels_
        return self.clusters_obj
    

    def DBSCAN_algorithm(self, eps, n_min, layer, scaled='no-scale', metric='euclidean'):
        if scaled == 'z-score':
            self.scaled = 'z-score'
            vectors = self.z_score(layer)
            self.vectors_z_score = vectors
        else:
            self.scaled = 'no-scale'
            vectors = self.vectors_per_layer[layer]

        if metric == 'cosine':
            cosine_sim = cosine_similarity(vectors)
            # Ajustar los valores de similitud de coseno cercanos a 1
            cosine_sim = np.clip(cosine_sim, a_min=0, a_max=1)
            dbscan = DBSCAN(eps=eps, min_samples=n_min, metric='precomputed')
            vectors = 1-cosine_sim
        elif metric == 'chebyshev':
            dbscan = DBSCAN(eps=eps, min_samples=n_min, metric='chebyshev')
        elif metric == 'mahalanobis':
            covariance_matrix = np.cov(vectors, rowvar=False)
            dbscan = DBSCAN(eps=eps, min_samples=n_min, metric='mahalanobis', metric_params={'V': covariance_matrix})
        elif metric == 'manhattan':
            dbscan = DBSCAN(eps=eps, min_samples=n_min, metric='minkowski', p=1)
        else:
            dbscan = DBSCAN(eps=eps, min_samples=n_min)
        self.clusters_obj = dbscan.fit(vectors)
        self._labels = self.clusters_obj.labels_
        return self.clusters_obj


    def mean_shift_algorithm(self, bandwidth, layer, scaled='no-scale'):
        if scaled == 'z-score':
            self.scaled = 'z-score'
            vectors = self.z_score(layer)
            self.vectors_z_score = vectors
        else:
            self.scaled = 'no-scale'
            vectors = self.vectors_per_layer[layer]

        meanshift = MeanShift(bandwidth=bandwidth)
        self.clusters_obj = meanshift.fit(vectors)

        self._labels = self.clusters_obj.labels_
        cluster_centers = meanshift.cluster_centers_  # Obtiene los centroides
        # Calcula la inercia como la suma de las distancias al cuadrado entre los puntos y sus centroides
        inertia = np.sum((vectors - cluster_centers[self._labels])**2)
        return self.clusters_obj, inertia
    

    def agglomerative_algorithm(self, k, layer, scaled='no-scale', metric_='euclidean', linkage_='ward'):
        if scaled == 'z-score':
            self.scaled = 'z-score'
            vectors = self.z_score(layer)
            self.vectors_z_score = vectors
        else:
            self.scaled = 'no-scale'
            vectors = self.vectors_per_layer[layer]

        if linkage_=='ward' and metric_ !='euclidean':
            messagebox.showerror("Error", "Linkage ward solo acepta distancia euclidiana")
        agglomerative = AgglomerativeClustering(n_clusters=k, metric=metric_, linkage=linkage_)
        self.clusters_obj = agglomerative.fit(vectors)
        self._labels = self.clusters_obj.labels_
        # Obtiene los centroides de los clústeres
        cluster_centers = [vectors[self._labels == i].mean(axis=0) for i in range(k)]
        # Calcula la inercia manualmente
        inertia = sum(np.sum((vectors[self._labels == i] - cluster_centers[i]) ** 2) for i in range(k))
        return self.clusters_obj, inertia
    

    def spectral_algorithm(self, k, layer, scaled='no-scale', metric='nearest_neighbors', n_neighbors_=10, gamma_=1.0):
        if scaled == 'z-score':
            self.scaled = 'z-score'
            vectors = self.z_score(layer)
            self.vectors_z_score = vectors
        else:
            self.scaled = 'no-scale'
            vectors = self.vectors_per_layer[layer]

        spectral = SpectralClustering(n_clusters=k, eigen_solver='arpack', affinity=metric, n_neighbors=n_neighbors_, gamma=gamma_, n_init=20, random_state=0)
        self.clusters_obj = spectral.fit(vectors)
        self._labels = self.clusters_obj.labels_
        return self.clusters_obj
    

    def gaussian_mixture_algorithm(self, k, layer, scaled='no-scale', covariance='full'):
        if scaled == 'z-score':
            self.scaled = 'z-score'
            vectors = self.z_score(layer)
            self.vectors_z_score = vectors
        else:
            self.scaled = 'no-scale'
            vectors = self.vectors_per_layer[layer]

        gaussian_mixture = GaussianMixture(n_components=k, covariance_type=covariance, n_init=30, random_state=0)
        gaussian_mixture.fit(vectors)
        self.probs_gaussians = gaussian_mixture.predict_proba(vectors)
        self._labels = gaussian_mixture.predict(vectors)
        data = {
            'probs_': self.probs_gaussians,
            'aic': gaussian_mixture.aic(vectors),
            'bic': gaussian_mixture.bic(vectors)
        }
        self.clusters_obj = ResultFormat(self._labels, data)
        return self.clusters_obj


    def gridSearch(self, param_grid, layer, scaled='no-scale', algorithm_type='spectral_clustering'):
        if scaled == 'z-score':
            self.scaled = 'z-score'
            vectors = self.z_score(layer)
            self.vectors_z_score = vectors
        else:
            self.scaled = 'no-scale'
            vectors = self.vectors_per_layer[layer]

        # Crear una instancia 
        #if algorithm_type == 'spectral_clustering':
        algorithm = SpectralClustering()

        # Definir una función de puntuación basada en la métrica de silueta
        def silhouette_scorer(estimator, X):
            labels = estimator.fit_predict(X)
            score = silhouette_score(X, labels)
            return score

        # Realizar la búsqueda de hiperparámetros con validación cruzada usando la puntuación de silueta
        grid_search = GridSearchCV(estimator=algorithm, param_grid=param_grid, cv=5, scoring=silhouette_scorer)
        grid_search.fit(vectors)  # data es tu conjunto de datos

        # Obtener los mejores hiperparámetros
        return grid_search.best_params_



    def get_vectors(self, layer):
        if self.scaled == 'z-score':
            return self.vectors_z_score
        elif self.scaled == 'no-scale':
            return self.vectors_per_layer[layer]

    def silhouetteScoreMetric(self, layer):
        return silhouette_score(self.get_vectors(layer), self._labels)
    
    def davisBouldinMetric(self, layer):
        return davies_bouldin_score(self.get_vectors(layer), self._labels)
    
    def calinskiMetric(self, layer):
        return calinski_harabasz_score(self.get_vectors(layer), self._labels)
    
    def get_labels(self, isround):
        if isround:
           return [round(number) for number in list(self.labels)]
        else:
            return self.labels
    
    def ARIMetric(self, isround):
        return adjusted_rand_score(self.get_labels(isround), self._labels)
    
    def rand_Metric(self, isround):
        return rand_score(self.get_labels(isround), self._labels)
    
    def homogeneity_completeness_v_measureMetric(self, isround):
        h,c,v = homogeneity_completeness_v_measure(self.get_labels(isround), self._labels)
        return (h,c,v)
    
    def mutual_infoMetric(self, isround):
        return mutual_info_score(self.get_labels(isround), self._labels)
    
    def fowlkes_mallowsMetric(self, isround):
        return fowlkes_mallows_score(self.get_labels(isround), self._labels)


class ResultFormat:
    def __init__(self, labels, data):
        self.labels_ = labels
        self.data = data