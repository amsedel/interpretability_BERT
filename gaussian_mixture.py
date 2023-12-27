import numpy as np
from tkinter import messagebox
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from plotting import *
from clustering_algorithms import ClusteringAlgorithms
from plot_interface import Plot_Interface


class Gaussian_Mixture:
    def __init__(self, instances):
        self.ins = instances
        self.layer = int(instances['entry_layer'].get())
        #self.type_analysis = self.ins['radio_clus_ana_var'].get()
        self.type_analysis = self.ins['radio_embedding_type_analysis'].get()
        self.method = self.ins['radio_var'].get()
        self.data = ClusteringAlgorithms(self.type_analysis, self.method)
        self.gaussian_mixture = self.data

    def update_data(self):
        self.type_analysis = self.ins['radio_embedding_type_analysis'].get()
        self.method = self.ins['radio_var'].get()
        self.data = ClusteringAlgorithms(self.type_analysis, self.method)
        self.gaussian_mixture = self.data
        if self.method == '1':
            self.layer = int(self.ins['pca'].layer)
        else:
            self.layer = int(self.ins['entry_layer'].get())

    def view_gaussian_mixture(self):
        label_k = tk.Label(self.ins['window'], text="Define K: ", font="Arial 14")
        label_k.place(relx=0.07, rely=0.24, anchor=tk.CENTER)
        self.entry_k = tk.Entry(self.ins['window'], width=8)
        self.entry_k.place(relx=0.065, rely=0.27, anchor=tk.CENTER)

        btn_k = tk.Button(self.ins['window'], text="Sugerir K", command=self.ins['go_to_calculate_gaussian_mixture'])
        btn_k.place(relx=0.065, rely=0.32, anchor=tk.CENTER)

        plot_gaussian_mixture = Plot_Interface(self.ins,[1.2],(6,6))
        plot_gaussian_mixture.view_scale_radios()
        plot_gaussian_mixture.view_metric_radios(default_metrics='Covarianza completa', metrics = {'Covarianza completa': {'rx':0.065,'ry':0.11,'tag':'full'},'Covarianza ligada':{'rx':0.065,'ry':0.14,'tag':'tied'},'Covarianza diagonal':{'rx':0.065,'ry':0.17,'tag':'diag'},'Covarianza esférica':{'rx':0.065,'ry':0.2,'tag':'spherical'}})
        pos_radio = {'lx1':0.07, 'ly1':0.37,'rx1':0.018,'ry1':0.4,'rx2':0.018,'ry2':0.43,'rx3':0.018,'ry3':0.46,'rx4':0.018,'ry4':0.49}
        plot_gaussian_mixture.view_radios_btn({'view': self.ins['window'], 'fn':lambda:None}, pos_radio, type_='clustering')
        plot_gaussian_mixture.view_btn("Mezcla Gaussiana", self.gaussianMixture, config = {'x': 0.065, 'y':0.55})
        plot_gaussian_mixture.view_btns_options(pos = {'bx1':0.064,'by1':0.60,'bx2':0.064, 'by2':0.65, 'bx3': 0.064, 'by3':0.7})
        plot_gaussian_mixture.view_plot()

        self.ins['plot'] = plot_gaussian_mixture.plot
        self.ins['canvas'] = plot_gaussian_mixture.canvas
        self.ins['canvas_widget'] =plot_gaussian_mixture.canvas_widget
        self.ins['zoom_factor'] =plot_gaussian_mixture.zoom_factor
        self.ins['zoom_direction_button'] = plot_gaussian_mixture.btn_zoom_direction
        self.ins['radio_var_label_plot'] = plot_gaussian_mixture.radio_var_label_plot
        self.ins['var-norm'] = plot_gaussian_mixture.var_norm
        self.ins['var-metric'] = plot_gaussian_mixture.var_metric


    def view_calculate_gaussian_mixture(self):
        label_k_cal = tk.Label(self.ins['calculate_gaussian_mixture_window'], text="K máximo: ", font="Arial 14")
        label_k_cal.place(relx=0.335, rely=0.075, anchor=tk.CENTER)

        clus_k_cal = tk.IntVar(self.ins['calculate_gaussian_mixture_window'], 10)
        self.entry_k_cal  = tk.Entry(self.ins['calculate_gaussian_mixture_window'], textvariable=clus_k_cal, width=10)
        self.entry_k_cal.place(relx=0.665, rely=0.075, anchor=tk.CENTER)

        btn_ = tk.Button(self.ins['calculate_gaussian_mixture_window'], text="Estimar K", command=self.calculate_k)
        btn_.place(relx=0.5, rely=0.15, anchor=tk.CENTER)

        self.best_k_sil = tk.Label(self.ins['calculate_gaussian_mixture_window'], text=" ", font="Arial 16")
        self.best_k_sil.place(relx=0.5, rely=0.215, anchor=tk.CENTER)


    def gaussianMixture(self):
        self.update_data()
        k = int(self.entry_k.get())
        text = text_options(self.ins)
        self.scaled = self.ins['var-norm'].get()
        if self.scaled == 'z-score':
            X = self.data.z_score(self.layer)
        else:
            X = self.data.vectors_per_layer[self.layer]
        self.gaussian_mixture = self.data.gaussian_mixture_algorithm(k, self.layer, scaled=self.scaled, covariance=self.ins['var-metric'].get())
        plot_clustering(self.ins, X, self.gaussian_mixture.labels_, 'Mezcla gausiana', self.data.labels, text)


    def view_plot(self, n_components, values, config={'ylabel':'','xlabel':'','title':'','x':50,'y':200,'w':350,'h':350}):
        fig = Figure(figsize=(5, 4), dpi=100)
        plot = fig.add_subplot(111)
        plot.plot(n_components, values, marker='o')
        plot.set_xlabel(config['xlabel'])
        plot.set_ylabel(config['ylabel'])
        plot.set_title(config['title'])
        canvas = FigureCanvasTkAgg(fig, master=self.ins['calculate_gaussian_mixture_window'])
        canvas.get_tk_widget().place(x=config['x'], y=config['y'], width=config['w'], height=config['h'])


    def calculate_k(self):
        self.update_data()
        try: 
            k_ = list(range(1,int(self.entry_k_cal.get())+1))
            aic_values, bic_values = [], []
            best_score = -1
            best_k = None
            for k in k_:
                result = self.data.gaussian_mixture_algorithm(k, self.layer, scaled=self.ins['var-norm'].get(), covariance=self.ins['var-metric'].get())
                aic_values.append(result.data['aic'])
                bic_values.append(result.data['bic'])
                if len(set(result.labels_)) > 1:  # Evitar divisiones con un solo clúster
                    score = self.data.silhouetteScoreMetric(self.layer)
                    if score > best_score:
                        best_score = score
                        best_k = k
            
            self.view_plot(k_,aic_values,config={'ylabel':'AIC','xlabel':'Número de componentes o clusters (k)','title':'Criterio de Información de Akaike (AIC)','x':50,'y':200,'w':500,'h':500})
            self.view_plot(k_,bic_values,config={'ylabel':'BIC','xlabel':'Número de componentes o clusters (k)','title':'Criterio de Información Bayesiano (BIC)','x':600,'y':200,'w':500,'h':500})

            self.best_k_sil.config(text=f"De acuerdo con silhouette_score se recomiendan {str(best_k)} clusters")

        except ValueError:
            messagebox.showerror("Error", "Asegurate de introducer un valor k entero en la caja de texto." )
        except Exception as e:
            messagebox.showerror("Error", "Se produjo una excepción:" + str(e))
