import numpy as np
from tkinter import messagebox
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from plotting import *
from clustering_algorithms import ClusteringAlgorithms
from plot_interface import Plot_Interface


class K_means:
    def __init__(self, instances):
        self.ins = instances
        self.layer = int(instances['entry_layer'].get())
        #self.type_analysis = self.ins['radio_clus_ana_var'].get()
        self.type_analysis = self.ins['radio_embedding_type_analysis'].get()
        self.method = self.ins['radio_var'].get()
        self.data = ClusteringAlgorithms(self.type_analysis, self.method)
        self.kmeans = self.data

    def update_data(self):
        self.type_analysis = self.ins['radio_embedding_type_analysis'].get()
        self.method = self.ins['radio_var'].get()
        self.data = ClusteringAlgorithms(self.type_analysis, self.method)
        self.kmeans = self.data
        if self.method == '1':
            self.layer = int(self.ins['pca'].layer)
        else:
            self.layer = int(self.ins['entry_layer'].get())

    def view_kmeans(self):
        label_k = tk.Label(self.ins['window'], text="Define K: ", font="Arial 14")
        label_k.place(relx=0.07, rely=0.2, anchor=tk.CENTER)
        self.entry_k = tk.Entry(self.ins['window'], width=8)
        self.entry_k.place(relx=0.065, rely=0.23, anchor=tk.CENTER)

        btn_k = tk.Button(self.ins['window'], text="Sugerir K", command=self.ins['go_to_calculate_k'])
        btn_k.place(relx=0.065, rely=0.28, anchor=tk.CENTER)

        plot_k_means = Plot_Interface(self.ins,[1.2],(6,6))
        plot_k_means.view_scale_radios()
        plot_k_means.view_metric_radios()
        plot_k_means.view_btn("K-means", self.k_means, config = {'x': 0.065, 'y':0.57})
        plot_k_means.view_btns_options(pos = {'bx1':0.064,'by1':0.65,'bx2':0.064, 'by2':0.7, 'bx3': 0.064, 'by3':0.75})
        pos_radio = {'lx1':0.07, 'ly1':0.35,'rx1':0.018,'ry1':0.4,'rx2':0.018,'ry2':0.435,'rx3':0.018,'ry3':0.47,'rx4':0.018,'ry4':0.505}
        plot_k_means.view_radios_btn({'view': self.ins['window'], 'fn':lambda:None}, pos_radio, type_='clustering')
        plot_k_means.view_plot()

        self.ins['plot'] = plot_k_means.plot
        self.ins['canvas'] = plot_k_means.canvas
        self.ins['canvas_widget'] =plot_k_means.canvas_widget
        self.ins['zoom_factor'] =plot_k_means.zoom_factor
        self.ins['zoom_direction_button'] = plot_k_means.btn_zoom_direction
        self.ins['radio_var_label_plot'] = plot_k_means.radio_var_label_plot
        self.ins['var-norm'] = plot_k_means.var_norm
        self.ins['var-metric'] = plot_k_means.var_metric

    def view_calculate_k(self):
        label_k_elbow = tk.Label(self.ins['calculate_k_window'], text="Valor de K hasta donde deseas probar: ", font="Arial 14")
        label_k_elbow.place(relx=0.26, rely=0.075, anchor=tk.CENTER)

        clus_k_elbow = tk.IntVar(self.ins['calculate_k_window'], 10)
        self.entry_k_elbow  = tk.Entry(self.ins['calculate_k_window'], textvariable=clus_k_elbow)
        self.entry_k_elbow.place(relx=0.64, rely=0.075, anchor=tk.CENTER)
        btn_k_elbow = tk.Button(self.ins['calculate_k_window'], text="Calcular K", command=self.calculate_k)
        btn_k_elbow.place(relx=0.89, rely=0.074, anchor=tk.CENTER)

        self.best_k_sil = tk.Label(self.ins['calculate_k_window'], text=" ", font="Arial 16")
        self.best_k_sil.place(relx=0.5, rely=0.15, anchor=tk.CENTER)


    def k_means(self):
        self.update_data()
        k = int(self.entry_k.get())
        text = text_options(self.ins)
        self.scaled = self.ins['var-norm'].get()
        if self.scaled == 'z-score':
            X = self.data.z_score(self.layer)
        else:
            X = self.data.vectors_per_layer[self.layer]
        self.kmeans = self.data.k_means_algorithm(k, self.layer, scaled=self.scaled, metric=self.ins['var-metric'].get())
        #kmeans = KMeans(n_clusters=k, random_state=0, n_init=30)
        #clusters_k_means = kmeans.fit_predict(X)
        plot_clustering(self.ins, X, self.kmeans.labels_, 'k-means', self.data.labels, text)


    def view_single_plot(self, k_, inercias):
        fig = Figure(figsize=(5, 4), dpi=100)
        plot = fig.add_subplot(111)
        plot.plot(range(1, k_), inercias, marker='o')
        plot.set_xlabel('Número de clústeres (k)')
        plot.set_ylabel('Inercia')
        plot.set_title('Método Elbow')
        canvas = FigureCanvasTkAgg(fig, master=self.ins['calculate_k_window'])
        canvas.get_tk_widget().place(x=100, y=120, width=400, height=400)

        #k means
    def calculate_k(self):
        self.update_data()
        inercias = []
        silueta_scores = []
        try: 
            k_ = int(self.entry_k_elbow.get())
            for k in range(1, k_):
                kmeans = self.data.k_means_algorithm(k, self.layer, scaled=self.ins['var-norm'].get(), metric=self.ins['var-metric'].get())
                inercias.append(kmeans.inertia_)
                if k > 1:
                    silueta_scores.append(self.data.silhouetteScoreMetric(self.layer))
            self.view_single_plot(k_, inercias)    
            best_k_silueta = np.argmax(silueta_scores) + 2
            self.best_k_sil.config(text=f"De acuerdo con silhouette_score se recomiendan {str(best_k_silueta)} clusters")
        except ValueError:
            messagebox.showerror("Error", "Asegurate de introducer un valor k entero en la caja de texto." )
        except Exception as e:
            messagebox.showerror("Error", "Se produjo una excepción:" + str(e))
