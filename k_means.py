import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import silhouette_samples, silhouette_score
from plotting import *
from clustering_algorithms import ClusteringAlgorithms


def get_type_analysis(event, instances):
    global type_analysis
    type_analysis = instances['radio_clus_ana_var'].get()
    global data, vectors_per_layer, sequences, labels
    data = ClusteringAlgorithms(type_analysis)
    vectors_per_layer, sequences, labels = data.vectors_per_layer, data.sequences, data.labels


#k means
def calculate_k(instances):
    inercias = []
    silueta_scores = []
    try: 
        k_ = int(instances['entry_k_elbow'].get())
        layer = int(instances['entry_clus_layer'].get())
        for k in range(1, k_):
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(vectors_per_layer[layer])
            inercias.append(kmeans.inertia_)

            if k > 1:
                kmeanss = KMeans(n_clusters=k, random_state=0)
                labels_ = kmeanss.fit_predict(vectors_per_layer[layer])
                silueta = silhouette_score(vectors_per_layer[layer], labels_)
                silueta_scores.append(silueta)
        fig = Figure(figsize=(5, 4), dpi=100)
        plot = fig.add_subplot(111)
        plot.plot(range(1, k_), inercias, marker='o')
        plot.set_xlabel('Número de clústeres (k)')
        plot.set_ylabel('Inercia')
        plot.set_title('Método Elbow')
        canvas = FigureCanvasTkAgg(fig, master=instances['window'])
        canvas.get_tk_widget().place(x=100, y=120, width=400, height=400)
        best_k_silueta = np.argmax(silueta_scores) + 2
        instances['best_k_sil'].config(text=f"De acuerdo con silhouette_score se recomiendan {str(best_k_silueta)} clusters")
    except ValueError:
        instances['messagebox'].showerror("Error", "Asegurate de introducer un valor k entero en la caja de texto." )
    except Exception as e:
        instances['messagebox'].showerror("Error", "Se produjo una excepción:" + str(e))


def k_means(instances):
    layer = int(instances['entry_clus_layer'].get())
    k = int(instances['entry_k'].get())
    X = vectors_per_layer[layer]
    text = text_options(instances)
    kmeans = data.k_means_algorithm(k, layer)
    #kmeans = KMeans(n_clusters=k, random_state=0, n_init=30)
    #clusters_k_means = kmeans.fit_predict(X)
    plot_clustering(instances, X, kmeans.labels_, 'k-means', labels, text)
