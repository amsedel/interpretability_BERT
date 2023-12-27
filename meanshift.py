import numpy as np
from tkinter import messagebox
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from plotting import *
from clustering_algorithms import ClusteringAlgorithms
from plot_interface import Plot_Interface


class Mean_Shift:
    def __init__(self, instances):
        self.ins = instances
        self.layer = int(instances['entry_layer'].get())
        #self.type_analysis = self.ins['radio_clus_ana_var'].get()
        self.type_analysis = self.ins['radio_embedding_type_analysis'].get()
        self.method = self.ins['radio_var'].get()
        self.data = ClusteringAlgorithms(self.type_analysis, self.method)
        self.meanshift = self.data

    def update_data(self):
        self.type_analysis = self.ins['radio_embedding_type_analysis'].get()
        self.method = self.ins['radio_var'].get()
        self.data = ClusteringAlgorithms(self.type_analysis, self.method)
        self.meanshift = self.data
        if self.method == '1':
            self.layer = int(self.ins['pca'].layer)
        else:
            self.layer = int(self.ins['entry_layer'].get())

    def view_meanshift(self):

        label_bandwidth = tk.Label(self.ins['window'], text="Define Bandwidth: ", font="Arial 14")
        label_bandwidth.place(relx=0.07, rely=0.2, anchor=tk.CENTER)
        self.entry_bandwidth = tk.Entry(self.ins['window'], width=8)
        self.entry_bandwidth.place(relx=0.065, rely=0.23, anchor=tk.CENTER)

        btn_bandwidth = tk.Button(self.ins['window'], text="Sugerir Bandwidth", command=self.ins['go_to_calculate_bandwidth'])
        btn_bandwidth.place(relx=0.065, rely=0.28, anchor=tk.CENTER)

        plot_mean_shift = Plot_Interface(self.ins,[1.2],(6,6))
        plot_mean_shift.view_scale_radios()
        plot_mean_shift.view_btn("Mean-shift", self.mean_shift, config = {'x': 0.065, 'y':0.57})
        plot_mean_shift.view_btns_options(pos = {'bx1':0.064,'by1':0.65,'bx2':0.064, 'by2':0.7, 'bx3': 0.064, 'by3':0.75})
        pos_radio = {'lx1':0.07, 'ly1':0.35,'rx1':0.018,'ry1':0.4,'rx2':0.018,'ry2':0.435,'rx3':0.018,'ry3':0.47,'rx4':0.018,'ry4':0.505}
        plot_mean_shift.view_radios_btn({'view': self.ins['window'], 'fn':lambda:None}, pos_radio, type_='clustering')
        plot_mean_shift.view_plot()

        self.ins['plot'] = plot_mean_shift.plot
        self.ins['canvas'] = plot_mean_shift.canvas
        self.ins['canvas_widget'] =plot_mean_shift.canvas_widget
        self.ins['zoom_factor'] =plot_mean_shift.zoom_factor
        self.ins['zoom_direction_button'] = plot_mean_shift.btn_zoom_direction
        self.ins['radio_var_label_plot'] = plot_mean_shift.radio_var_label_plot
        self.ins['var-norm'] = plot_mean_shift.var_norm

    def view_calculate_bandwidth(self):
        label_bandwidth_elbow = tk.Label(self.ins['calculate_bandwidth_window'], text="Introduce valores Bandwidth: ", font="Arial 14")
        label_bandwidth_elbow.place(relx=0.25, rely=0.05, anchor=tk.CENTER)

        clus_bandwidth_elbow = tk.StringVar(self.ins['calculate_bandwidth_window'], '0.001,0.005,0.01,0.02,0.03')
        self.entry_bandwidth_elbow  = tk.Entry(self.ins['calculate_bandwidth_window'], textvariable=clus_bandwidth_elbow)
        self.entry_bandwidth_elbow.place(relx=0.7, rely=0.05, anchor=tk.CENTER)

        btn_bandwidth_elbow = tk.Button(self.ins['calculate_bandwidth_window'], text="Calcular bandwidth", command=self.calculate_bandwidth)
        btn_bandwidth_elbow.place(relx=0.5, rely=0.11, anchor=tk.CENTER)

        self.best_bandwidth_sil = tk.Label(self.ins['calculate_bandwidth_window'], text=" ", font="Arial 16")
        self.best_bandwidth_sil.place(relx=0.5, rely=0.18, anchor=tk.CENTER)


    def mean_shift(self):
        self.update_data()
        bandwidth = float(self.entry_bandwidth.get())
        text = text_options(self.ins)
        self.scaled = self.ins['var-norm'].get()
        if self.scaled == 'z-score':
            X = self.data.z_score(self.layer)
        else:
            X = self.data.vectors_per_layer[self.layer]
        self.meanshift, _ = self.data.mean_shift_algorithm(bandwidth, self.layer, scaled=self.scaled)
        plot_clustering(self.ins, X, self.meanshift.labels_, 'Mean-shift', self.data.labels, text)

    def view_single_plot(self, bandwidths, inercias):
        fig = Figure(figsize=(5, 4), dpi=100)
        plot = fig.add_subplot(111)
        plot.plot(bandwidths, inercias, marker='o')
        plot.set_xlabel('Bandwidth')
        plot.set_ylabel('Inercia')
        plot.set_title('Método Elbow')
        canvas = FigureCanvasTkAgg(fig, master=self.ins['calculate_bandwidth_window'])
        canvas.get_tk_widget().place(x=100, y=135, width=400, height=400)

        #k means
    def calculate_bandwidth(self):
        self.update_data()
        inercias = []
        silueta_scores = []
        try: 
            bandwidths = self.entry_bandwidth_elbow.get().split(',')
            for b in bandwidths:
                meanshifts, inertia = self.data.mean_shift_algorithm(float(b), self.layer, scaled=self.ins['var-norm'].get())
                inercias.append(inertia)
                clusters = list(set(meanshifts.labels_))
                if len(clusters) > 1:
                    silueta_scores.append(self.data.silhouetteScoreMetric(self.layer))
            self.view_single_plot(bandwidths, inercias)    
            best_bandwidth_index = np.argmax(silueta_scores)
            best_bandwidth = bandwidths[best_bandwidth_index]
            self.best_bandwidth_sil.config(text=f"De acuerdo con silhouette_score se recomienda un bandwidth de {str(best_bandwidth)}")
        except ValueError:
            messagebox.showerror("Error", "Asegurate de introducer un valor valido en la caja de texto o reducir el valor máximo de bandwidth." )
        except Exception as e:
            messagebox.showerror("Error", "Se produjo una excepción:" + str(e))
