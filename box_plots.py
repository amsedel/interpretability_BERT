import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn
import scipy.stats as stats
from scipy.stats import friedmanchisquare
from sklearn.utils import resample


class Box_Plot:
    def __init__(self, instances):
        self.instances = instances
        self.layer = int(self.instances['entry_layer'].get())
        self.linguistic_instance = self.instances['linguistic_analysis']
        self.title = ''
        self.titles = []
        self.box_data = []
        self.ylabel = ''
        self.all_cluster_data = []
        self.all_data = []
        self.heat_maps_list = []


    def create_dataframe(self):
        self.linguistic_instance = self.instances['linguistic_analysis']
        self.data = {'vectors': list(self.linguistic_instance.current_instance.data.vectors_per_layer[self.layer]),
                'sequences': list(self.linguistic_instance.current_instance.data.sequences.values()),
                'labels': list(self.linguistic_instance.current_instance.data.labels.values()),
                'dimensions': list(self.linguistic_instance.current_instance.data.dimensions.values()),
                'cluster_labels': self.linguistic_instance.current_instance.data._labels,
                'structures_labels': self.linguistic_instance.structures_labels}
        self.data['all_samples']= list(np.zeros(len(self.data['vectors']), dtype=int))
        data = self.data
        if len(self.data['cluster_labels']) == 0:
            self.data['cluster_labels'] = list(np.zeros(len(self.data['vectors']), dtype=int))
        if len(self.data['structures_labels']) == 0:
            self.data['structures_labels'] = list(np.zeros(len(self.data['vectors']), dtype=int))
        if self.linguistic_instance.algorithm_name == 'DBSCAN' and -1 in self.data['cluster_labels']:
            max_num = max(self.data['cluster_labels']) + 1
            ##max_num = max(self.data['cluster_labels'])
            replace_minus_one = [i if i != -1 else max_num for i in self.data['cluster_labels']]
            data = self.data
            data['cluster_labels'] = replace_minus_one

        return pd.DataFrame(data), data


    def view_box_plot(self):
        btn_ = tk.Button(self.instances['window'], text="Box plot", command=self.view_plot)
        btn_.grid(row=0, column=0, padx=10, pady=10)
        btn_save = tk.Button(self.instances['window'], text="Guardar", command=self.save_plot)
        btn_save.grid(row=0, column=1, padx=10, pady=10)
        btn_see = tk.Button(self.instances['window'], text="Ver todo", command=self.instances['go_to_view_all_plots'])
        btn_see.grid(row=0, column=2, padx=10, pady=10)
        btn_see_eval = tk.Button(self.instances['window'], text="Evaluar", command=self.evaluate)
        btn_see_eval.grid(row=0, column=3, padx=10, pady=10)
        btn_save_img = tk.Button(self.instances['window'], text="Guardar Img", command=self.save_img_plot)
        btn_save_img.grid(row=0, column=4, padx=10, pady=10)
        self.threshold = tk.StringVar(self.instances['window'], "0.0001")
        self.entry_threshold = tk.Entry(self.instances['window'], width=8, textvariable=self.threshold)
        self.entry_threshold.grid(row=0, column=5, padx=10, pady=10)



    def view_all_box_plot(self):
        self.radio_var = tk.IntVar(self.instances['view_all_plots_window'], -1)
        # Crear radio buttons
        radio1 = tk.Radiobutton(self.instances['view_all_plots_window'], text="Box plots", variable=self.radio_var, value=1, command=self.all_plots)
        radio2 = tk.Radiobutton(self.instances['view_all_plots_window'], text="Dispersión", variable=self.radio_var, value=2, command=self.all_plots)
        radio3 = tk.Radiobutton(self.instances['view_all_plots_window'], text="DispersiónClus", variable=self.radio_var, value=3, command=self.all_plots)
        radio4 = tk.Radiobutton(self.instances['view_all_plots_window'], text="Anotaciones", variable=self.radio_var, value=4, command=self.all_plots)
        radio5 = tk.Radiobutton(self.instances['view_all_plots_window'], text="Leyendas", variable=self.radio_var, value=5, command=self.all_plots)
        radio6 = tk.Radiobutton(self.instances['view_all_plots_window'], text="HeatMaps", variable=self.radio_var, value=6, command=self.all_plots)
        radio7 = tk.Radiobutton(self.instances['view_all_plots_window'], text="HeatMapsValues", variable=self.radio_var, value=7, command=self.all_plots)
        radio8 = tk.Radiobutton(self.instances['view_all_plots_window'], text="HeatMapsMark", variable=self.radio_var, value=8, command=self.all_plots)
        save_plot = tk.Button(self.instances['view_all_plots_window'], text="Guardar ImgPlot", command=self.save_img_plot)
        radio1.grid(row=0, column=0)
        radio2.grid(row=1, column=0)
        radio3.grid(row=2, column=0)
        radio4.grid(row=3, column=0)
        radio5.grid(row=4, column=0)
        radio6.grid(row=5, column=0)
        radio7.grid(row=6, column=0)
        radio8.grid(row=7, column=0)
        save_plot.grid(row=8, column=0, padx=10, pady=10)

    def save_img_plot(self):
        self.ax_.savefig(self.plot_name, dpi=500, bbox_inches='tight')

    def base_statistics_len_sequences(self):
        media = np.mean(np.array(self.data['dimensions']))
        median = np.median(np.array(self.data['dimensions']))
        variance = np.var(np.array(self.data['dimensions']))
        des_std = np.std(np.array(self.data['dimensions']))
        print("media: ", media)
        print("mediana: ", median)
        print("variance: ", variance)
        print("desviación estandar: ", des_std)


    def save_plot(self):
        self.titles.append(self.title)
        self.box_data.append(self.cluster_data)
        self.all_data.append(self.data['vectors'])
        self.all_cluster_data.append(self.data['cluster_labels'])
        self.heat_maps_list.append(self.posthoc_dunn_results)


    def all_plots(self):
        if str(self.radio_var.get()) == "1":
            data_ = self.box_data
        elif str(self.radio_var.get()) == "6" or str(self.radio_var.get()) == "7" or str(self.radio_var.get()) == "8":
            data_ = self.heat_maps_list
        else:
            data_ = list(self.linguistic_instance.current_instance.data.vectors_per_layer.values())

        fig = Figure(figsize=(4, 4))
        fig.subplots_adjust(hspace=0.62, wspace=0.4)
        #fig.subplots_adjust(hspace=0.62, wspace=0.4, left=0.1, right=0.9, top=0.9, bottom=0.1)
        fig.tight_layout()
        for i, data in enumerate(data_[:len(self.all_data)]):
            ax = fig.add_subplot(4, 3, i+1)
            if str(self.radio_var.get()) == "1":
                ax.set_xlabel('Agrupamientos', fontsize=8)
                #ax.set_xlabel('Clusters', fontsize=8)
                ax.set_ylabel(self.ylabel, fontsize=8)
                ax.set_title(f'{self.titles[i]}', fontsize=10)
                ax.grid(True)
                ax.tick_params(axis='x', labelsize=6)
                ax.tick_params(axis='y', labelsize=6)
                outlier_marker = dict(marker='o', markersize=4)
                ax.boxplot(data, vert=True, patch_artist=True, flierprops=outlier_marker)
                self.ax_ = fig
                self.plot_name = 'box_plot_full.png'
                    # Calcular la media de los datos
                medians = [np.mean(dataset) for dataset in data]
                # Dibujar la línea de la media en el diagrama de caja
                #ax.scatter(range(1, len(data) + 1), medians, color='red', marker='*', zorder=3)
            elif str(self.radio_var.get()) == "2":
                ax.set_title(f'Capa {i + 1}', fontsize=11)
                #ax.set_title(f'Layer {i + 1}', fontsize=11)
                ax.scatter(data[:, 0], data[:, 1], color='blue', s=1)
                self.ax_ = fig
                self.plot_name = 'dispersion_full.png'
            elif str(self.radio_var.get()) == "3" or str(self.radio_var.get()) == "4" or str(self.radio_var.get()) == "5" :
                ax.set_title(f'{self.titles[i]}', fontsize=11)
                clusters_list = sorted(list(set(self.all_cluster_data[i])))
                n_clusters = len(np.unique(self.all_cluster_data[i]))
                legends = [str(i+1) if i != -1 else 'Valores atípicos' for i in clusters_list]
                #legends = [str(i+1) if i != -1 else 'Atypical values' for i in clusters_list]
                clusters_ = [i if i != -1 else max(clusters_list)+1 for i in self.all_cluster_data[i]]
                clusters_list = [i if i != -1 else max(clusters_list)+1 for i in clusters_list]
                # Obtén una paleta de colores para asignar a los clusters
                cmap = cm.get_cmap('tab20', n_clusters)   
                for i, cluster in enumerate(clusters_list):
                    cluster_points = data[clusters_ == cluster]
                    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], color=cmap(cluster), label=f'c{legends[i]}', s=1)
                    self.ax_ = fig
                    self.plot_name = 'dispersin_full.png'
                    if str(self.radio_var.get()) == "4":
                        cm_ = np.mean(cluster_points, axis=0)
                        ax.annotate(f'{i+1}', (cm_[0], cm_[1]), fontsize=7, textcoords="offset points", xytext=(0,0), ha='center')
                        self.ax_ = fig
                        self.plot_name = 'dispersion_full_annotation.png'
                    if str(self.radio_var.get()) == "5":
                        ax.legend(loc='upper left', fontsize='small')
                        self.ax_ = fig
                        self.plot_name = 'dispersion_full_legeds.png'
            elif str(self.radio_var.get()) == "6" or str(self.radio_var.get()) == "7" or str(self.radio_var.get()) == "8":
                cmap2 = plt.get_cmap('coolwarm')
                new_cmap = mcolors.ListedColormap(cmap2(np.linspace(1, 0, 512)))
                ax.set_title(f'Capa {i + 1}', fontsize=11)
                #ax.set_title(f'Layer {i + 1}', fontsize=11)
                heatmap = ax.imshow(data, cmap=new_cmap, interpolation='none')
                ax.set_xticks(np.arange(data.shape[1]))
                ax.set_yticks(np.arange(data.shape[0]))
                ax.set_xticklabels(np.arange(1, data.shape[1] + 1), fontsize=5)
                ax.set_yticklabels(np.arange(1, data.shape[0] + 1), fontsize=5)
                fig.colorbar(heatmap)
                self.ax_ = fig
                self.plot_name = 'dunn_test.png'
                if str(self.radio_var.get()) == "7":
                    for i in range(data.shape[0]):
                        for j in range(data.shape[1]): #data es un dataframe
                            ax.text(j, i, f'{data.values[i, j]:.2f}', va='center', ha='center', color='black', fontsize=5)
                    self.ax_ = fig
                    self.plot_name = 'dunn_test_values.png'
                if str(self.radio_var.get()) == "8":
                    for i in range(data.shape[0]):
                        for j in range(data.shape[1]): #data es un dataframe
                            if data.values[i, j] < float(self.entry_threshold.get()):
                                ax.text(j, i, '*', va='center', ha='center', color='black', fontsize=7)
                    self.ax_ = fig
                    self.plot_name = 'dunn_test_sign.png'

        # Crear un lienzo de Matplotlib para la figura
        canvas = FigureCanvasTkAgg(fig, master=self.instances['view_all_plots_window'])
        canvas.get_tk_widget().grid(row=0, column=1, rowspan=12, columnspan=12)
        #canvas.get_tk_widget().config(width=1200, height=350)
        canvas.get_tk_widget().config(width=850, height=800)
        

    def view_plot(self):
        self.layer = int(self.instances['entry_layer'].get())
        df, data = self.create_dataframe()
        #self.base_statistics_len_sequences()
        #xlabel = 'Clusters'
        xlabel = 'Agrupamientos'
        n_clusters = len(set(data['cluster_labels']))
        if self.linguistic_instance.analysis_type == "3":
            self.ylabel = 'Tamaño secuencias'
            #self.ylabel = 'Sequence length'
            data__ = [df[df['all_samples'] == i]['dimensions'] for i in range(1)]
            img_name = 'len_sequence.png'
            title_ = 'Distribución de dimensión de secuencias'
            #title_ = 'Sequence dimension distribution'
            self.cluster_data = [df[df['cluster_labels'] == i]['dimensions'] for i in range(n_clusters)]

        if self.linguistic_instance.analysis_type == "2":
            self.ylabel = 'Similitud gramatical'
            #self.ylabel = 'Grammatical similarity'
            data__ = [df[df['all_samples'] == i]['structures_labels'] for i in range(1)]
            img_name = 'sim_gram.png'
            title_ = 'Distribución de similitud estructural'
            #title_ = 'Structural similarity distribution'
            self.cluster_data = [df[df['cluster_labels'] == i]['structures_labels'] for i in range(n_clusters)]

        if self.linguistic_instance.analysis_type == "4":
            self.ylabel = 'Etiquetas'
            #self.ylabel = 'Labels'
            data__ = [df[df['all_samples'] == i]['labels'] for i in range(1)]
            img_name = 'labels.png'
            title_ = 'Distribución de etiquetas'
            #title_ = 'Label distribution'
            self.cluster_data = [df[df['cluster_labels'] == i]['labels'] for i in range(n_clusters)]

        self.title = 'Capa ' + str(self.layer+1)
        #self.title = 'Layer ' + str(self.layer+1)
        self.plotting(xlabel, self.ylabel, self.title, self.cluster_data)
        config={'r':3,'c':10,'w':500,'h':500}
        self.plotting('Conjunto de datos', self.ylabel, title_, data__, config, img_name)
        #self.plotting('Test dataset', self.ylabel, title_, data__, config, img_name)




    def plotting(self, xlabel, ylabel, title, data, config={'r':3,'c':0,'w':500,'h':500}, img_name = ''):
        # Crear una figura de matplotlib
        fig = Figure(figsize=(4, 4), dpi=100)
        ax = fig.add_subplot(111)
        # Dibujar el boxplot en el eje
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ###ax.set_title('Distribución del tamaño de las secuencias')
        ax.boxplot(data, vert=True, patch_artist=True)
        ax.grid(True)
        ax.tick_params(axis='x', labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        medians = [np.mean(dataset) for dataset in data]
        # Dibujar la línea de la media en el diagrama de caja
        ax.scatter(range(1, len(data) + 1), medians, color='black', marker='*', zorder=3)
        self.ax_ = fig
        self.plot_name = img_name
        # Crear un lienzo de tkinter para el boxplot
        canvas = FigureCanvasTkAgg(fig, master=self.instances['window'])
        canvas_widget = canvas.get_tk_widget()
        # Usar el sistema de cuadrícula para colocar el lienzo en la ventana
        canvas_widget.grid(row=config['r'], column=config['c'], padx=10, pady=10, rowspan=8, columnspan=8)
        canvas_widget.config(width=config['w'], height=config['h'])         


    def evaluate(self):
        threshold = float(self.entry_threshold.get())
        if self.linguistic_instance.analysis_type == "3":
            variable = np.array(self.data['dimensions'])
        if self.linguistic_instance.analysis_type == "2":
            variable = np.array(self.data['structures_labels'])
        if self.linguistic_instance.analysis_type == "4":
            variable = np.array(self.data['labels'])

        # Obtén las clusters únicas
        n_clusters = np.unique(self.data['cluster_labels'])
        groups = {}

        # Itera a través de las clusters únicas y agrupa las variable
        for label in n_clusters:
            indices = np.where(self.data['cluster_labels'] == label)[0]
            variable_cluster = variable[indices]
            #self.stats_(variable_cluster,label)      
            groups[label] = variable_cluster
        
        print('*****Shapiro-Levene*****'*5)
        #pruebas de normalidad (Shapiro-Wilk) y pruebas de igualdad de varianzas (Levene) para determinar si tenemos grupos paramétricos o no paramétricos
        for i, grupo in enumerate(list(groups.values())):
            self.shapiro_test(grupo, i)
        self.levene_test(list(groups.values()))

        n_groups = len(list(groups.values()))
        if n_groups == 2:
            _, p, test_type = self.mannwhitneyu(list(groups.values()))
        else:
            _, p, test_type = self.kruskal_test(list(groups.values()))


        if n_groups > 2:
            alpha = threshold / n_groups
        else: 
            alpha = threshold

        if p < alpha:
            print(f"Hay diferencias significativas entre los grupos para {test_type} con {alpha}.")
        if p > alpha:
            print(f"No hay diferencias significativas entre los grupos con {alpha}.")
        self.posthoc_dunn_results = self.dunn_test(list(groups.values()))



    def stats_(self,variable_cluster,label):
        print("Cluster " + str(label))
        media = np.mean(variable_cluster)
        print(f"Media: {media}")
        varianza = np.var(variable_cluster, ddof=1)  # El argumento ddof=1 indica que se debe usar la fórmula de la varianza muestral
        print(f"Varianza muestral: {varianza}")
        desviacion_estandar = np.std(variable_cluster, ddof=1)  # El argumento ddof=1 indica que se debe usar la fórmula de la desviación estándar muestral
        print(f"Desviación estándar muestral: {desviacion_estandar}")


    def shapiro_test(self, variable_cluster, group_name):
        # Imprimir resultados de la prueba de Shapiro-Wilk
        print("Pruebas de Shapiro-Wilk para normalidad:")
        stat, pvalue = stats.shapiro(variable_cluster)
        if pvalue > 0.05:
            print(f"Grupo {group_name}: Los datos parecen seguir una distribución normal (p={pvalue:.4f})")
        else:
            print(f"Grupo {group_name}: Los datos no siguen una distribución normal (p={pvalue:.4f})")

    def levene_test(self, grupos):
        levene_stat, levene_pvalue = stats.levene(*grupos)
        # Imprimir resultado de la prueba de Levene
        print("\nPrueba de Levene para igualdad de varianzas:")
        if levene_pvalue > 0.05:
            print("No hay evidencia de diferencias significativas en las varianzas (p=",levene_pvalue,")")
        else:
            print("Hay evidencia de diferencias significativas en las varianzas (p=",levene_pvalue,")")
        print()


    def dunn_test(self, data):
        posthoc_results = posthoc_dunn(data, p_adjust = 'holm')
        print('-*-'*20)
        print("Resultados de la prueba post hoc de Dunn: ")
        print(posthoc_results)
        return posthoc_results

    def friedman_test(self, groups):
        stat, p = stats.f_oneway(*groups)
        print("-FRIEDMAN-"*10)
        print("Estadística de FRIEDMAN:", stat)
        print("Valor p FRIEDMAN:", p)
        return stat, p
    

    def welch_anova_test(self, groups):
        stat, p = stats.f_oneway(*groups)
        print("-WA-"*30)
        print("Estadística de Welch-ANOVA:", stat)
        print("Valor p Welch_ANOVA:", p)
        return stat, p
    
    def kruskal_test(self, groups):
        stat, p = stats.kruskal(*groups)
        print("-K-"*30)
        print("Estadística de Kruskal:", stat)
        print("Valor p de Kruskal:", p)
        return stat, p, 'Kruskal Wallis'

    def mannwhitneyu(self, groups):
        stat, p_value = stats.mannwhitneyu(*groups)
        print('-MWU-'*20)
        print("Resultados de la prueba Mann Whitney U: ")
        print("Estadística de Mann Whitney U:", stat)
        print("Valor p de Mann Whitney U:", p_value)
        return stat, p_value, 'Mann Whitney U'