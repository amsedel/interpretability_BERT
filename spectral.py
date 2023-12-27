import numpy as np
from tkinter import messagebox
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from plotting import *
from clustering_algorithms import ClusteringAlgorithms
from plot_interface import Plot_Interface


class Spectral:
    def __init__(self, instances):
        self.ins = instances
        self.layer = int(instances['entry_layer'].get())
        #self.type_analysis = self.ins['radio_clus_ana_var'].get()
        self.type_analysis = self.ins['radio_embedding_type_analysis'].get()
        self.method = self.ins['radio_var'].get()
        self.data = ClusteringAlgorithms(self.type_analysis, self.method)
        self.spectral = self.data

    def update_data(self):
        self.type_analysis = self.ins['radio_embedding_type_analysis'].get()
        self.method = self.ins['radio_var'].get()
        self.data = ClusteringAlgorithms(self.type_analysis, self.method)
        self.spectral = self.data
        if self.method == '1':
            self.layer = int(self.ins['pca'].layer)
        else:
            self.layer = int(self.ins['entry_layer'].get())

    def view_spectral(self):
        label_gamma = tk.Label(self.ins['window'], text="Gamma: ", font="Arial 14")
        label_gamma.place(relx=0.07, rely=0.18, anchor=tk.CENTER)
        var_gamma = tk.StringVar(self.ins['window'], 1.0)
        self.entry_gamma = tk.Entry(self.ins['window'], width=8, textvariable=var_gamma)
        self.entry_gamma.place(relx=0.065, rely=0.21, anchor=tk.CENTER)

        label_n_neigh = tk.Label(self.ins['window'], text="N-neighbors: ", font="Arial 14")
        label_n_neigh.place(relx=0.07, rely=0.25, anchor=tk.CENTER)
        var_n_neigh_elbow = tk.IntVar(self.ins['window'], 10)
        self.entry_n_neigh = tk.Entry(self.ins['window'], width=8, textvariable=var_n_neigh_elbow)
        self.entry_n_neigh.place(relx=0.065, rely=0.28, anchor=tk.CENTER)

        label_k = tk.Label(self.ins['window'], text="Define K: ", font="Arial 14")
        label_k.place(relx=0.07, rely=0.32, anchor=tk.CENTER)
        self.entry_k = tk.Entry(self.ins['window'], width=8)
        self.entry_k.place(relx=0.065, rely=0.35, anchor=tk.CENTER)

        btn_k = tk.Button(self.ins['window'], text="Sugerir Valores", command=self.ins['go_to_calculate_spectral'])
        btn_k.place(relx=0.065, rely=0.4, anchor=tk.CENTER)

        plot_spectral = Plot_Interface(self.ins,[1.2],(6,6))
        plot_spectral.view_scale_radios()
        plot_spectral.view_metric_radios(default_metrics='Vecinos más cercanos', metrics = {'Vecinos más cercanos': {'rx':0.065,'ry':0.11,'tag':'nearest_neighbors'},'Kernel RBF':{'rx':0.065,'ry':0.14,'tag':'rbf'}})
        pos_radio = {'lx1':0.07, 'ly1':0.45,'rx1':0.018,'ry1':0.48,'rx2':0.018,'ry2':0.515,'rx3':0.018,'ry3':0.55,'rx4':0.018,'ry4':0.585}
        plot_spectral.view_radios_btn({'view': self.ins['window'], 'fn':lambda:None}, pos_radio, type_='clustering')
        plot_spectral.view_btn("Spectral", self.spectral_clus, config = {'x': 0.065, 'y':0.65})
        plot_spectral.view_btns_options(pos = {'bx1':0.064,'by1':0.70,'bx2':0.064, 'by2':0.75, 'bx3': 0.064, 'by3':0.8})
        plot_spectral.view_plot()

        self.ins['plot'] = plot_spectral.plot
        self.ins['canvas'] = plot_spectral.canvas
        self.ins['canvas_widget'] =plot_spectral.canvas_widget
        self.ins['zoom_factor'] =plot_spectral.zoom_factor
        self.ins['zoom_direction_button'] = plot_spectral.btn_zoom_direction
        self.ins['radio_var_label_plot'] = plot_spectral.radio_var_label_plot
        self.ins['var-norm'] = plot_spectral.var_norm
        self.ins['var-metric'] = plot_spectral.var_metric

    def view_calculate_spectral(self):
        label_k_cal = tk.Label(self.ins['calculate_spectral_window'], text="K máximo: ", font="Arial 14")
        label_k_cal.place(relx=0.26, rely=0.075, anchor=tk.CENTER)

        clus_k_cal = tk.IntVar(self.ins['calculate_spectral_window'], 10)
        self.entry_k_cal  = tk.Entry(self.ins['calculate_spectral_window'], textvariable=clus_k_cal, width=10)
        self.entry_k_cal.place(relx=0.64, rely=0.075, anchor=tk.CENTER)

        btn_ = tk.Button(self.ins['calculate_spectral_window'], text="Estimar hiperparámetros", command=self.calculate_k)
        btn_.place(relx=0.5, rely=0.15, anchor=tk.CENTER)

        self.best_k_sil = tk.Label(self.ins['calculate_spectral_window'], text=" ", font="Arial 16")
        self.best_k_sil.place(relx=0.5, rely=0.25, anchor=tk.CENTER)
        self.best_measure_sil = tk.Label(self.ins['calculate_spectral_window'], text=" ", font="Arial 16")
        self.best_measure_sil.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

        self.label_n_neigh_cal = tk.Label(self.ins['calculate_spectral_window'], text="Número de vecinos: ", font="Arial 14")
        self.label_n_neigh_cal.place_forget()
        self.clus_n_neigh_cal = tk.StringVar(self.ins['calculate_spectral_window'], '5,10')
        self.entry_n_neigh_cal  = tk.Entry(self.ins['calculate_spectral_window'], textvariable=self.clus_n_neigh_cal, width=10)
        self.entry_n_neigh_cal.place_forget()
        self.btn_n = tk.Button(self.ins['calculate_spectral_window'], text="Estimar número de vecinos", command=self.define_hyperparameter)
        self.btn_n.place_forget()

        self.label_gamma_cal = tk.Label(self.ins['calculate_spectral_window'], text="Valores Gamma: ", font="Arial 14")
        self.label_gamma_cal.place_forget()
        self.clus_gamma_cal = tk.StringVar(self.ins['calculate_spectral_window'], '1.0,2.0')
        self.entry_gamma_cal  = tk.Entry(self.ins['calculate_spectral_window'], textvariable=self.clus_gamma_cal, width=10)
        self.entry_gamma_cal.place_forget()
        self.btn_gamma = tk.Button(self.ins['calculate_spectral_window'], text="Estimar gamma", command=self.define_hyperparameter)
        self.btn_gamma.place_forget()

        self.best_value = tk.Label(self.ins['calculate_spectral_window'], text=" ", font="Arial 16")
        self.best_value.place(relx=0.5, rely=0.65, anchor=tk.CENTER)


    def spectral_clus(self):
        self.update_data()
        k = int(self.entry_k.get())
        text = text_options(self.ins)
        self.scaled = self.ins['var-norm'].get()
        n_neigh = int(self.entry_n_neigh.get())
        gamma = float(self.entry_gamma.get())
        if self.scaled == 'z-score':
            X = self.data.z_score(self.layer)
        else:
            X = self.data.vectors_per_layer[self.layer]
        self.spectral = self.data.spectral_algorithm(k, self.layer, scaled=self.scaled, metric=self.ins['var-metric'].get(), n_neighbors_=n_neigh, gamma_=gamma)
        plot_clustering(self.ins, X, self.spectral.labels_, 'Spectral', self.data.labels, text)


        #k means
    def calculate_k(self):
        self.update_data()
        try: 
            k_ = list(range(2,int(self.entry_k_cal.get())+1))
            # Definir el rango de hiperparámetros a ajustar
            param_grid = {
                'n_clusters': k_,
                'affinity': ['nearest_neighbors', 'rbf']
            }
            best_result = self.data.gridSearch(param_grid, self.layer, scaled=self.ins['var-norm'].get(), algorithm_type='spectral_clustering')
            self.n_cluster = int(best_result['n_clusters'])
            self.measure = str(best_result['affinity'])

            self.best_k_sil.config(text=f"De acuerdo con silhouette_score se recomiendan {str(self.n_cluster)} clusters")
            self.best_measure_sil.config(text=f"Con una medida {self.measure}")

            if best_result['affinity'] == 'nearest_neighbors':
                self.label_n_neigh_cal.place(relx=0.26, rely=0.4, anchor=tk.CENTER)
                self.entry_n_neigh_cal.place(relx=0.64, rely=0.4, anchor=tk.CENTER)
                self.btn_n.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
            elif best_result['affinity'] == 'rbf':
                self.label_gamma_cal.place(relx=0.26, rely=0.4, anchor=tk.CENTER)
                self.entry_gamma_cal.place(relx=0.64, rely=0.4, anchor=tk.CENTER)
                self.btn_gamma.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        except ValueError:
            messagebox.showerror("Error", "Asegurate de introducer un valor k entero en la caja de texto." )
        except Exception as e:
            messagebox.showerror("Error", "Se produjo una excepción:" + str(e))


    def define_hyperparameter(self):
        self.update_data()
        if self.measure == 'nearest_neighbors':
            n_neigh = [int(i) for i in self.entry_n_neigh_cal.get().split(',')]
            param_grid = {
                'n_clusters': [self.n_cluster],
                'affinity': [self.measure],
                'n_neighbors': n_neigh
            }
            best_result = self.data.gridSearch(param_grid, self.layer, scaled=self.ins['var-norm'].get(), algorithm_type='spectral_clustering')
            self.best_value.config(text=f"Con mejor número de vecinos de : {str(best_result['n_neighbors'])}")
        elif self.measure == 'rbf':
            gamma = [float(i) for i in self.entry_gamma_cal.get().split(',')]
            param_grid = {
                'n_clusters': [self.n_cluster],
                'affinity': [self.measure],
                'gamma': gamma
            }
            best_result = self.data.gridSearch(param_grid, self.layer, scaled=self.ins['var-norm'].get(), algorithm_type='spectral_clustering')
            self.best_value.config(text=f"Con un valor gamma : {str(best_result['gamma'])}")