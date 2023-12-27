import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from clustering_algorithms import ClusteringAlgorithms
import numpy as np

class Sweeping_K:
    def __init__(self, instances):
        self.instances = instances
        self.layer = int(self.instances['entry_layer'].get())
        self.clusters = self.instances['kmeans']

    def validate_algorithm(self):
        self.method = self.instances['radio_var'].get()
        self.type_analysis = self.instances['radio_embedding_type_analysis'].get()
        self.layer = int(self.instances['entry_layer'].get())
        if self.method == '1':
            self.algorithm = self.instances['pca'].radio_clus_var.get()
        else: 
            self.algorithm = self.instances['radio_clus_var'].get()

        if self.algorithm == "1":
            self.clusters = self.instances['kmeans']
            self.alg_ins = self.instances['kmeans_instance']
        if self.algorithm == "2":
            self.clusters = self.instances['dbscan']
            self.alg_ins = self.instances['dbscan_instance']
        if self.algorithm == "3":
            self.clusters = self.instances['meanshift']
            self.alg_ins = self.instances['meanshift_instance']
        if self.algorithm == "4":
            self.clusters = self.instances['agglomerative']
            self.alg_ins = self.instances['agglomerative_instance']
        if self.algorithm == "5":
            self.clusters = self.instances['spectral']
            self.alg_ins = self.instances['spectral_instance']
        if self.algorithm == "6":
            self.clusters = self.instances['gaussian_mixture']
            self.alg_ins = self.instances['gaussian_mixture_instance']

    def focus_in(self):
        self.validate_algorithm()
        if self.algorithm == "2":
            self.k_cal.set('0.01,0.02,0.03')
            self.label_k.config(text="eps: ", font="Arial 14")
        elif self.algorithm == "3":
            self.k_cal.set('0.02,0.021,0.022')
            self.label_k.config(text="Bandwidths: ", font="Arial 14")
        else: 
            self.k_cal.set(10)
            self.label_k.config(text="K máximo: ", font="Arial 14")

    def view_sweeping_k(self):
        self.validate_algorithm()
        self.label_k= tk.Label(self.instances['window'], text=" ", font="Arial 14")
        self.label_k.grid(row=0, column=0, padx=10, pady=10)
        self.k_cal = tk.StringVar(self.instances['window'], ' ')
        self.entry_k = tk.Entry(self.instances['window'], textvariable=self.k_cal)
        self.entry_k.grid(row=0, column=1, padx=10, pady=10)
        btn_ = tk.Button(self.instances['window'], text="Calcular", command=self.calculate)
        btn_.grid(row=1, column=0, padx=10, pady=10)


    def kmeans_(self, k_list):
        silhouettes, dvs, calis, BICs, AICs, inertias  = [], [], [], [], [], []
        self.clus = ClusteringAlgorithms(self.type_analysis, self.method)
        for k in k_list:
            sca = self.clusters.scaled
            metric_ = self.alg_ins.ins['var-metric'].get()
            kmeans= self.clus.k_means_algorithm(k, self.layer, scaled=sca, metric=metric_)
            sil = self.clus.silhouetteScoreMetric(self.layer)
            dv = self.clus.davisBouldinMetric(self.layer)
            cali = self.clus.calinskiMetric(self.layer)
            inertias.append(kmeans.inertia_) 
            silhouettes.append(sil)
            calis.append(cali)
            dvs.append(dv)       
        return [np.array(silhouettes), np.array(dvs), np.array(calis), np.array(BICs), np.array(AICs), np.array(inertias)] 


    def dbscan_(self, eps):
        silhouettes, dvs, calis, BICs, AICs, inertias  = [], [], [], [], [], []
        self.clus = ClusteringAlgorithms(self.type_analysis, self.method)
        for k in eps:
            sca = self.clusters.scaled
            metric_ = self.alg_ins.ins['var-metric'].get()
            min_samples = int(self.alg_ins.entry_nmin.get()) 
            self.clus.DBSCAN_algorithm(k, min_samples, self.layer, scaled=sca, metric=metric_)
            sil = self.clus.silhouetteScoreMetric(self.layer)
            dv = self.clus.davisBouldinMetric(self.layer)
            cali = self.clus.calinskiMetric(self.layer)
            silhouettes.append(sil)
            calis.append(cali)
            dvs.append(dv)       
        return [np.array(silhouettes), np.array(dvs), np.array(calis), np.array(BICs), np.array(AICs), np.array(inertias)] 


    def meanshift_(self, k_list):
        silhouettes, dvs, calis, BICs, AICs, inertias  = [], [], [], [], [], []
        self.clus = ClusteringAlgorithms(self.type_analysis, self.method)
        for k in k_list:
            sca = self.clusters.scaled
            _, inertia = self.clus.mean_shift_algorithm(k, self.layer, scaled=sca)
            sil = self.clus.silhouetteScoreMetric(self.layer)
            dv = self.clus.davisBouldinMetric(self.layer)
            cali = self.clus.calinskiMetric(self.layer)
            inertias.append(inertia) 
            silhouettes.append(sil)
            calis.append(cali)
            dvs.append(dv)       
        return [np.array(silhouettes), np.array(dvs), np.array(calis), np.array(BICs), np.array(AICs), np.array(inertias)] 


    def agglomerative_(self, k_list):
        silhouettes, dvs, calis, BICs, AICs, inertias  = [], [], [], [], [], []
        self.clus = ClusteringAlgorithms(self.type_analysis, self.method)
        for k in k_list:
            sca = self.clusters.scaled
            metric = self.alg_ins.ins['var-metric'].get()
            linkage=self.alg_ins.ins['var-linkage'].get()
            _, inertia = self.clus.agglomerative_algorithm(k, self.layer, scaled=sca, metric_=metric, linkage_=linkage)
            sil = self.clus.silhouetteScoreMetric(self.layer)
            dv = self.clus.davisBouldinMetric(self.layer)
            cali = self.clus.calinskiMetric(self.layer)
            inertias.append(inertia) 
            silhouettes.append(sil)
            calis.append(cali)
            dvs.append(dv)       
        return [np.array(silhouettes), np.array(dvs), np.array(calis), np.array(BICs), np.array(AICs), np.array(inertias)] 

    def spectral_(self, k_list):
        silhouettes, dvs, calis, BICs, AICs, inertias  = [], [], [], [], [], []
        self.clus = ClusteringAlgorithms(self.type_analysis, self.method)
        for k in k_list:
            sca = self.clusters.scaled
            metric_ = self.alg_ins.ins['var-metric'].get()
            n_neigh = int(self.alg_ins.entry_n_neigh.get())
            gamma = float(self.alg_ins.entry_gamma.get())
            self.clus.spectral_algorithm(k, self.layer, scaled=sca, metric=metric_, n_neighbors_=n_neigh, gamma_=gamma)
            sil = self.clus.silhouetteScoreMetric(self.layer)
            dv = self.clus.davisBouldinMetric(self.layer)
            cali = self.clus.calinskiMetric(self.layer)
            silhouettes.append(sil)
            calis.append(cali)
            dvs.append(dv)       
        return [np.array(silhouettes), np.array(dvs), np.array(calis), np.array(BICs), np.array(AICs), np.array(inertias)]        


    def gaussian_mixture_(self, k_list):
        silhouettes, dvs, calis, BICs, AICs, inertias  = [], [], [], [], [], []
        self.clus = ClusteringAlgorithms(self.type_analysis, self.method)
        for k in k_list:
            sca = self.clusters.scaled
            cov = self.alg_ins.ins['var-metric'].get()
            self.clus.gaussian_mixture_algorithm(k, self.layer, scaled=sca, covariance=cov)
            sil = self.clus.silhouetteScoreMetric(self.layer)
            dv = self.clus.davisBouldinMetric(self.layer)
            cali = self.clus.calinskiMetric(self.layer)
            bic = self.clus.clusters_obj.data['bic']
            aic = self.clus.clusters_obj.data['aic']
            silhouettes.append(sil)
            BICs.append(bic)
            AICs.append(aic)
            calis.append(cali)
            dvs.append(dv)       

        return [np.array(silhouettes), np.array(dvs), np.array(calis), np.array(BICs), np.array(AICs), np.array(inertias)]


    def calculate(self):
        n=0
        self.validate_algorithm()
        if self.algorithm == "2" or self.algorithm == "3":
            k_list = [float(i) for i in self.entry_k.get().split(',')]
        else:
            if '-' in self.entry_k.get():
                range_ = self.entry_k.get().split('-')
                n = int(range_[0])
                k_list = list(range(int(range_[0]),int(range_[1])+1))
            else:
                n = 2
                k_list = list(range(n,int(self.entry_k.get())+1))
        fig = Figure(figsize=(4, 4))
        data = []
        # Crear múltiples subplots en la figura
        subplots = [fig.add_subplot(231), fig.add_subplot(232), fig.add_subplot(233),
                    fig.add_subplot(234), fig.add_subplot(235), fig.add_subplot(236)]
        titles = ['Índice de Silueta', 'Índice de Davies-Bouldin', 'Índice de Calinski Harabasz', 'BIC', 'AIC','Inercia']
        xlabel = 'K'
        if self.algorithm == "1":
            data = self.kmeans_(k_list)
        elif self.algorithm == "2":
            data = self.dbscan_(k_list)
            xlabel = 'eps'
        elif self.algorithm == "3":
            data = self.meanshift_(k_list)
            xlabel = 'bandwidth'
        if self.algorithm == "4":
            data = self.agglomerative_(k_list)
        elif self.algorithm == "5":
            data = self.spectral_(k_list)
        elif self.algorithm == "6":
            data = self.gaussian_mixture_(k_list)

        ks = k_list
        k_ops, k_values = self.get_best_values(data, ks, n, titles)
        titles_abrev = ['Silueta', 'Davies-Bouldin', 'Calinski', 'BIC', 'AIC','Inercia']
        self.result_table(k_ops, k_values, xlabel, titles_abrev)
        fig.subplots_adjust(hspace=0.35, wspace=0.45)
        # Personalizar y agregar datos a cada gráfica
        for i, ax in enumerate(subplots):
            ax.set_xlabel(xlabel)
            ax.set_ylabel(titles[i])
            ax.set_title(titles[i])
            if len(data[i]) == 0:
                ax.plot([], data[i], marker='o')
            else:
                ax.plot(ks, data[i], marker='o')
            if k_ops[i] != None:
                ax.axvline(x=k_ops[i], color='red', linestyle='--', label=f'k óptimo= {k_ops[i]}')
                ax.legend()

        # Crear un lienzo de Matplotlib para la figura
        canvas = FigureCanvasTkAgg(fig, master=self.instances['window'])
        canvas.get_tk_widget().grid(row=0, column=4, padx=10, pady=10, rowspan=12, columnspan=12)
        canvas.get_tk_widget().config(width=1150, height=700)


    def get_best_values(self, data, k_list, n, titles):
        k_ops = []
        values = []
        for i, t in enumerate(titles): 
            if len(data[i]) == 0 or t == 'Inercia':
                k_op = None
                value = None
            else:
                if self.algorithm != '2' and self.algorithm != '3':
                    if t == 'Índice de Silueta' or t == 'Índice de Calinski Harabasz':
                        value = np.amax(data[i])
                        k_op = np.argmax(data[i]) + n
                    elif t == 'Índice de Davies-Bouldin' or t=='BIC' or t=='AIC':
                        value = np.amin(data[i])
                        k_op = np.argmin(data[i]) + n
                else:
                    if t == 'Índice de Silueta' or t == 'Índice de Calinski Harabasz':
                        value = np.amax(data[i])
                        k_op = k_list[np.argmax(data[i])]
                    elif t == 'Índice de Davies-Bouldin' or t=='BIC' or t=='AIC':
                        value = np.amin(data[i])
                        k_op = k_list[np.argmin(data[i])]
            k_ops.append(k_op)
            values.append(value)
        return k_ops, values


    def result_table(self, k_ops, k_values, xlabel, titles):
        # Crear un Treeview (Tabla)
        table = ttk.Treeview(self.instances['window'], columns=("Métrica", "Mejor valor", "# K"), show="headings")

        # Definir las columnas
        table.heading("#1", text="Métrica")
        table.heading("#2", text="Mejor valor")
        table.heading("#3", text=xlabel)

        # Agregar datos
        for i, t in enumerate(titles):
            if t != 'Inercia':
                table.insert("", "end", text="",  values=(t, k_values[i], k_ops[i]))

        # Alineación de las columnas
        table.column("#1", anchor="w")
        table.column("#2", anchor="center")
        table.column("#3", anchor="w")

        table.column("#1", minwidth=75, width=80)
        table.column("#2", minwidth=45, width=60)
        table.column("#3", minwidth=10, width=60)
        # Colocar la tabla en la ventana utilizando grid
        table.grid(row=3, column=0, padx=10, pady=10, columnspan=3)

