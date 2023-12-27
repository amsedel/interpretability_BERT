import numpy as np
from tkinter import messagebox
import tkinter as tk
from plotting import *
from clustering_algorithms import ClusteringAlgorithms
from plot_interface import Plot_Interface


class Dbscan:
    def __init__(self, instances):
        self.ins = instances
        self.layer = int(instances['entry_layer'].get())
        #self.type_analysis = self.ins['radio_clus_ana_var'].get()
        self.type_analysis = self.ins['radio_embedding_type_analysis'].get()
        self.method = self.ins['radio_var'].get()
        self.data = ClusteringAlgorithms(self.type_analysis, self.method)
        self.dbscan = self.data

    def update_data(self):
        self.type_analysis = self.ins['radio_embedding_type_analysis'].get()
        self.method = self.ins['radio_var'].get()
        self.data = ClusteringAlgorithms(self.type_analysis, self.method)
        self.dbscan = self.data
        if self.method == '1':
            self.layer = int(self.ins['pca'].layer)
        else:
            self.layer = int(self.ins['entry_layer'].get())

    def view_dbscan(self):
        label_eps = tk.Label(self.ins['window'], text="Define eps (distancia): ", font="Arial 14")
        label_eps.place(relx=0.07, rely=0.27, anchor=tk.CENTER)
        self.entry_eps = tk.Entry(self.ins['window'], width=8)
        self.entry_eps.place(relx=0.065, rely=0.31, anchor=tk.CENTER)

        label_nmin = tk.Label(self.ins['window'], text="Define n_min (Vecinos): ", font="Arial 14")
        label_nmin.place(relx=0.07, rely=0.35, anchor=tk.CENTER)
        self.entry_nmin = tk.Entry(self.ins['window'], width=8)
        self.entry_nmin.place(relx=0.065, rely=0.39, anchor=tk.CENTER)

        btn_eps_nmin = tk.Button(self.ins['window'], text="Sugerir Valores", command=self.ins['go_to_calculate_eps_n_min'])
        btn_eps_nmin.place(relx=0.065, rely=0.44, anchor=tk.CENTER)

        plot_dbscan = Plot_Interface(self.ins,[1.2],(6,6))
        plot_dbscan.view_scale_radios()
        plot_dbscan.view_metric_radios(metrics = {'Euclidiana': {'rx':0.065,'ry':0.11,'tag':'euclidean'},'Coseno':{'rx':0.065,'ry':0.14,'tag':'cosine'},'Chebyshev':{'rx':0.065,'ry':0.17,'tag':'chebyshev'},'Mahalanobis':{'rx':0.065,'ry':0.20,'tag':'mahalanobis'},'Manhattan':{'rx':0.065,'ry':0.23,'tag':'manhattan'}})
        plot_dbscan.view_btn("DBSCAN", self.dbscan_method, config = {'x': 0.065, 'y':0.67})
        plot_dbscan.view_btns_options(pos = {'bx1':0.064,'by1':0.72,'bx2':0.064, 'by2':0.77, 'bx3': 0.064, 'by3':0.82})
        pos_radio = {'lx1':0.07, 'ly1':0.48,'rx1':0.018,'ry1':0.51,'rx2':0.018,'ry2':0.54,'rx3':0.018,'ry3':0.57,'rx4':0.018,'ry4':0.6}
        plot_dbscan.view_radios_btn({'view': self.ins['window'], 'fn':lambda:None}, pos_radio, type_='clustering')
        plot_dbscan.view_plot()

        self.ins['plot'] = plot_dbscan.plot
        self.ins['canvas'] = plot_dbscan.canvas
        self.ins['canvas_widget'] =plot_dbscan.canvas_widget
        self.ins['zoom_factor'] =plot_dbscan.zoom_factor
        self.ins['zoom_direction_button'] = plot_dbscan.btn_zoom_direction
        self.ins['radio_var_label_plot'] = plot_dbscan.radio_var_label_plot
        self.ins['var-norm'] = plot_dbscan.var_norm
        self.ins['var-metric'] = plot_dbscan.var_metric

    def view_calculate_dbscan(self):
        label_eps_elbow_from = tk.Label(self.ins['calculate_eps_nmin_window'], text="eps inicio: ", font="Arial 14")
        label_eps_elbow_from.place(relx=0.13, rely=0.12, anchor=tk.CENTER)
        clus_eps_elbow_from = tk.IntVar(self.ins['calculate_eps_nmin_window'], 0.1)
        self.entry_eps_elbow_from = tk.Entry(self.ins['calculate_eps_nmin_window'], width=8, textvariable=clus_eps_elbow_from)
        self.entry_eps_elbow_from.place(relx=0.26, rely=0.12, anchor=tk.CENTER)

        label_eps_elbow_to = tk.Label(self.ins['calculate_eps_nmin_window'], text="eps final: ", font="Arial 14")
        label_eps_elbow_to.place(relx=0.43, rely=0.12, anchor=tk.CENTER)
        clus_eps_elbow_to = tk.IntVar(self.ins['calculate_eps_nmin_window'], 0.5)
        self.entry_eps_elbow_to  = tk.Entry(self.ins['calculate_eps_nmin_window'], width=8, textvariable=clus_eps_elbow_to)
        self.entry_eps_elbow_to.place(relx=0.56, rely=0.12, anchor=tk.CENTER)

        label_div = tk.Label(self.ins['calculate_eps_nmin_window'], text="Incrementos: ", font="Arial 14")
        label_div.place(relx=0.73, rely=0.12, anchor=tk.CENTER)
        clus_div = tk.IntVar(self.ins['calculate_eps_nmin_window'], 10)
        self.entry_div  = tk.Entry(self.ins['calculate_eps_nmin_window'], width=8, textvariable=clus_div)
        self.entry_div.place(relx=0.85, rely=0.12, anchor=tk.CENTER)

        label_n_min_elbow_from = tk.Label(self.ins['calculate_eps_nmin_window'], text="n_min inicio: ", font="Arial 14")
        label_n_min_elbow_from.place(relx=0.25, rely=0.2, anchor=tk.CENTER)
        clus_n_min_elbow_from = tk.IntVar(self.ins['calculate_eps_nmin_window'], 1)
        self.entry_n_min_elbow_from  = tk.Entry(self.ins['calculate_eps_nmin_window'], width=8, textvariable=clus_n_min_elbow_from)
        self.entry_n_min_elbow_from.place(relx=0.4, rely=0.2, anchor=tk.CENTER)

        label_n_min_elbow_to = tk.Label(self.ins['calculate_eps_nmin_window'], text="n_min final: ", font="Arial 14")
        label_n_min_elbow_to.place(relx=0.6, rely=0.2, anchor=tk.CENTER)
        clus_n_min_elbow_to = tk.IntVar(self.ins['calculate_eps_nmin_window'], 10)
        self.entry_n_min_elbow_to  = tk.Entry(self.ins['calculate_eps_nmin_window'], width=8, textvariable=clus_n_min_elbow_to)
        self.entry_n_min_elbow_to.place(relx=0.75, rely=0.2, anchor=tk.CENTER)

        btn_elbow = tk.Button(self.ins['calculate_eps_nmin_window'], text="Calcular", command=self.calculate_eps_nmin)
        btn_elbow.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

        self.best_eps = tk.Label(self.ins['calculate_eps_nmin_window'], text=" ", font="Arial 16")
        self.best_eps.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.best_nmin = tk.Label(self.ins['calculate_eps_nmin_window'], text=" ", font="Arial 16")
        self.best_nmin.place(relx=0.5, rely=0.6, anchor=tk.CENTER)

    def dbscan_method(self):
        self.update_data()
        eps = float(self.entry_eps.get())
        n_min = int(self.entry_nmin.get())
        self.scaled = self.ins['var-norm'].get()
        if self.scaled == 'z-score':
            X = self.data.z_score(self.layer)
        else:
            X = self.data.vectors_per_layer[self.layer]
        text = text_options(self.ins)
        self.dbscan = self.data.DBSCAN_algorithm(eps, n_min, self.layer, scaled=self.scaled, metric=self.ins['var-metric'].get())
        plot_clustering(self.ins, X, self.dbscan.labels_, 'dbscan', self.data.labels, text)



    def calculate_eps_nmin(self):
        self.update_data()
        try:
            eps_i = float(self.entry_eps_elbow_from.get())
            eps_f = float(self.entry_eps_elbow_to.get())
            nmin_i = int(self.entry_n_min_elbow_from.get())
            nmin_f = int(self.entry_n_min_elbow_to.get())
            self.scaled = self.ins['var-norm'].get()
            div = int(self.entry_div.get())

            eps_values = np.linspace(eps_i, eps_f, num=div)
            min_samples_values = range(nmin_i, nmin_f + 1)
            best_score = -1
            best_eps = None
            best_min_samples = None

            for eps in eps_values:
                for min_samples in min_samples_values:
                    dbscan = self.data.DBSCAN_algorithm(eps, min_samples, self.layer, scaled=self.scaled, metric=self.ins['var-metric'].get())
                    labels = dbscan.labels_
                    if len(set(labels)) > 1:  # Evitar divisiones con un solo clúster
                        score = self.data.silhouetteScoreMetric(self.layer)
                        if score > best_score:
                            best_score = score
                            best_eps = eps
                            best_min_samples = min_samples
            if best_eps == None and best_min_samples == None:
                self.best_eps.config(text=f"De acuerdo con silhouette score se recomienda un eps de: {str(best_eps)}")
                self.best_nmin.config(text=f"De acuerdo con silhouette score se recomienda un n_min de: {str(best_min_samples)}")
            else:
                self.best_eps.config(text=f"De acuerdo con silhouette score se recomienda un eps de: {str(round(best_eps,4))}")
                self.best_nmin.config(text=f"De acuerdo con silhouette score se recomienda un n_min de: {str(best_min_samples)}")
            #print("Best Silhouette Score:", best_score)
            #print("Best eps:", best_eps)
            #print("Best min_samples:", best_min_samples)
        except ValueError:
            messagebox.showerror("Error", "Asegurate de introducer valores válidos." )
        except Exception as e:
            messagebox.showerror("Error", "Se produjo una excepción:" + str(e))
