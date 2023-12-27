
import tkinter as tk
from tkinter import messagebox
import numpy as np
import torch
import os
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



class PCA_:
    def __init__(self, instances):
        self.instances = instances


    def update_view(self):
        # Ocultar todos los conjuntos 
        self.set1.place_forget()
        self.set2.place_forget()
        self.set3.place_forget()
        self.layer = int(self.entry_layer.get())
        # Mostrar el set de botones correspondiente a la selección
        seleccion = self.radio_var.get()
        #self.set_instances()
        if seleccion == 1:
            self.set1.place(x=10, y=100)
            self.define_PCs_view()
        elif seleccion == 2:
            self.set2.place(x=10, y=100)
            self.biplot_view()
        elif seleccion == 3:
            self.set3.place(x=10, y=100)
            self.score_clustering_view()


    def pca_view(self):
        btn_pca_back = tk.Button(self.instances['window'], text="<-", command=self.instances['go_to_compression_window'])
        btn_pca_back.place(relx=0.05, rely=0.04, anchor=tk.CENTER)
        label_load_pth = tk.Label(self.instances['window'], text="Introduce la ruta del objeto .pth: ", font="Arial 14")
        label_load_pth.place(relx=0.2, rely=0.04, anchor=tk.CENTER)
        self.entry_root = tk.Entry(self.instances['window'], width=35)
        self.entry_root.place(relx=0.4, rely=0.04, anchor=tk.CENTER)
        btn_load_pth = tk.Button(self.instances['window'], text="Cargar .pth", command=self.load_pth)
        btn_load_pth.place(relx=0.56, rely=0.04, anchor=tk.CENTER)
        self.label_valid = tk.Label(self.instances['window'], text=" ", font="Arial 12", fg="GRAY")
        self.label_valid.place(relx=0.64, rely=0.04, anchor=tk.CENTER)
        layer_label = tk.Label(self.instances['window'], text="Capa: ", font="Arial 14")
        layer_label.place(relx=0.77, rely=0.04, anchor=tk.CENTER)
        self.layer= tk.IntVar(self.instances['window'], 0)
        self.entry_layer = tk.Entry(self.instances['window'], width=8, textvariable=self.layer)
        self.entry_layer.place(relx=0.85, rely=0.04, anchor=tk.CENTER)

        # Variable para almacenar la selección del radio button
        self.radio_var = tk.IntVar(self.instances['window'], -1)

        # Crear radio buttons
        radio1 = tk.Radiobutton(self.instances['window'], text="Definir PC's", variable=self.radio_var, value=1, command=self.update_view)
        radio2 = tk.Radiobutton(self.instances['window'], text="Biplot", variable=self.radio_var, value=2, command=self.update_view)
        radio3 = tk.Radiobutton(self.instances['window'], text="Clustering", variable=self.radio_var, value=3, command=self.update_view)

        # Ubicar los radio buttons en la ventana
        radio1.place(relx=0.25, rely=0.1, anchor=tk.CENTER)
        radio2.place(relx=0.5, rely=0.1, anchor=tk.CENTER)
        radio3.place(relx=0.75, rely=0.1, anchor=tk.CENTER)

        # Crear conjuntos de botones
        self.set1 = tk.Frame(self.instances['window'])
        self.set2 = tk.Frame(self.instances['window'])
        self.set3 = tk.Frame(self.instances['window'])

    def load_pth(self):
        root = self.entry_root.get()
        self.root_to_save = os.path.dirname(os.path.abspath(__file__))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.data = torch.load(root, map_location=torch.device(device))
            self.vectors_per_layer, self.sequences, self.labels, self.dimensions =  self.all_sequences_per_layer(self.instances['config'])
            self.data_per_layer = {
            'vectors': self.vectors_per_layer,
            'sequences': self.sequences,
            'labels': self.labels,
            'dimensions': self.dimensions
            }
            if self.instances['radio_embedding_type_analysis'].get() == 'Attention':
                torch.save(self.data_per_layer, self.root_to_save + '/data_per_layer_PCA_attention.pth')
            if self.instances['radio_embedding_type_analysis'].get() == 'CLS':
                torch.save(self.data_per_layer, self.root_to_save + '/data_per_layer_PCA_CLS.pth')
            self.label_valid.config(text=f"Archivo cargado.")
        except Exception as e:
            # Manejo de otras excepciones
            print(f"Error al cargar el archivo .pth: {e}")
            messagebox.showerror("Error", "Error al cargar el archivo " + root)
    
    def define_PCs_view(self):
        len_variables = self.vectors_per_layer[self.layer].shape[1]
        label_num_variables = tk.Label(self.set1, text=f"Se detectaron {len_variables} variables", font="Arial 14")
        label_num_variables.grid(row=0, column=0, padx=10, pady=10, columnspan=3, sticky="n")
        label_n_components = tk.Label(self.set1, text="n componentes: ", font="Arial 14")
        label_n_components.grid(row=1, column=0, padx=10, pady=10, sticky="n")
        var_n_components= tk.IntVar(self.set1, len_variables)
        self.entry_n_components = tk.Entry(self.set1, width=6, textvariable=var_n_components)
        self.entry_n_components.grid(row=1, column=1, padx=10, pady=10, sticky="n")
        self.btn_view = tk.Button(self.set1, text="Ver", command=self.pca_tools_view)
        self.btn_view.grid(row=1, column=2, padx=10, pady=10, sticky="n")

    def get_representations_per_layer(self, num_sentences, vector_representations, config):
        vectors_per_layer = {}
        labels = {}
        for l in range(config['layers']):
            #vectors_per_layer[l] = np.array([vector_representations[(i,l)]['vector'].detach().numpy() for i in range(num_sentences)])
            vectors_per_layer[l] = np.array([vector_representations[(i,l)]['vector'] for i in range(num_sentences)])
        labels = { i: vector_representations[(i,0)]['label'][0].item() for i in range(num_sentences)}
        sequences = { i: vector_representations[(i,0)]['sequence'][0] for i in range(num_sentences)}
        dimensions = { i: vector_representations[(i,0)]['dimension'][0].item() for i in range(num_sentences)}
        return vectors_per_layer, sequences, labels, dimensions


    def all_sequences_per_layer(self, config):
        num_sequences = len(self.data) // config['layers']
        return self.get_representations_per_layer(num_sequences, self.data, config)

    def pca_tools_view(self):
        n_components = int(self.entry_n_components.get())
        data_scaled = self.scaler_()
        pca_trans, pca = self.pca_(data_scaled, n_components)
        exp_var_r = pca.explained_variance_ratio_
        exp_var = pca.explained_variance_  
        PC_number = np.arange(pca.n_components_) + 1
        self.plotting('Componentes Principales', 'Proporción de varianza', 'Scree Plot (Método de codo)', {'x':PC_number, 'y':exp_var_r}, self.set1)
        config2={'r':3,'c':5,'w':440,'h':440}
        self.plotting('Componentes Principales', 'Porcentaje de Varianza Explicada', 'Porcentaje de Varianza Explicada por CP', {'x':PC_number, 'y':exp_var_r}, self.set1, config2, 'bar')
        config3={'r':3,'c':10,'w':440,'h':440}
        self.plotting('Componentes Principales', 'Varianza', 'Scree Plot (Regla de Kaiser)', {'x':PC_number, 'y':exp_var}, self.set1, config3)



    def plotting(self, xlabel, ylabel, title, data, ins, config={'r':3,'c':0,'w':440,'h':440}, plot_type='plot'):
        # Crear una figura de matplotlib
        fig = Figure(figsize=(4, 4), dpi=100)
        ax = fig.add_subplot(111)
        # Dibujar el boxplot en el eje
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=12)
        ###ax.set_title('Distribución del tamaño de las secuencias')
        if plot_type == 'plot':
            ax.plot(data['x'],data['y'],'ro-',color='blue')
            ax.grid(True)
            if 'Kaiser' in title:
                ax.axhline(y=1, color='r',linestyle='--')
        if plot_type == 'bar':
            ax.bar(range(len(data['y'])), data['y'], tick_label=data['x'], color='skyblue')
            ax.grid(False)
        ax.tick_params(axis='x', labelsize=7)
        ax.tick_params(axis='y', labelsize=7)

        canvas = FigureCanvasTkAgg(fig, master=ins)
        canvas_widget = canvas.get_tk_widget()
        # Usar el sistema de cuadrícula para colocar el lienzo en la ventana
        canvas_widget.grid(row=config['r'], column=config['c'], padx=10, pady=10, rowspan=5, columnspan=5)
        canvas_widget.config(width=config['w'], height=config['h'])      


    def scaler_(self):
        scaler = StandardScaler()
        return scaler.fit_transform(self.vectors_per_layer[self.layer]) 
    

    def pca_(self, data_scaled, n):
        # Ideal Number of Components
        pca = PCA(n_components = n) # Create PCA object
        pca_trans = pca.fit_transform(data_scaled)
        return pca_trans, pca


    def biplot_view(self):
        label_n_components = tk.Label(self.set2, text="n componentes: ", font="Arial 14")
        label_n_components.grid(row=0, column=0, padx=15, pady=10, sticky="n")
        var_n_components= tk.IntVar(self.set2, 2)
        self.entry_n_components_bi = tk.Entry(self.set2, width=6, textvariable=var_n_components)
        self.entry_n_components_bi.grid(row=0, column=1, padx=15, pady=10, sticky="n")

        label_PC1 = tk.Label(self.set2, text="PC(x): ", font="Arial 14")
        label_PC1.grid(row=2, column=0, padx=15, pady=10, sticky="n")
        pc1= tk.IntVar(self.set2, 1)
        self.entry_PC1 = tk.Entry(self.set2, width=5, textvariable=pc1)
        self.entry_PC1.grid(row=2, column=1, padx=15, pady=10, sticky="n")

        label_PC2 = tk.Label(self.set2, text="PC(y): ", font="Arial 14")
        label_PC2.grid(row=4, column=0, padx=15, pady=10, sticky="n")
        pc2= tk.IntVar(self.set2, 2)
        self.entry_PC2 = tk.Entry(self.set2, width=5, textvariable=pc2)
        self.entry_PC2.grid(row=4, column=1, padx=15, pady=10, sticky="n")

        self.btn_view = tk.Button(self.set2, text="Ver", command=self.pca_biplot_view)
        self.btn_view.grid(row=6, column=0, padx=15, pady=10, sticky="n")

    def save_pcs(self, pc1, pc2):
        data_per_layer = dict(self.data_per_layer)
        layers = list(range(0,int(self.instances['config']['layers'])))
        content = {}
        for i in layers:
            if self.layer != i:
                content[i] = np.zeros((len(pc1), 2))
            else: 
                content[i] = np.array((pc1,pc2)).T

        data_per_layer['vectors'] =  content
        if self.instances['radio_embedding_type_analysis'].get() == 'Attention':
            torch.save(data_per_layer, self.root_to_save + '/data_per_layer_PCA_attention.pth')
        if self.instances['radio_embedding_type_analysis'].get() == 'CLS':
            torch.save(data_per_layer, self.root_to_save + '/data_per_layer_PCA_CLS.pth')


    def pca_biplot_view(self):
        n_components = int(self.entry_n_components_bi.get())
        data_scaled = self.scaler_()
        pc, pca = self.pca_(data_scaled, n_components)
        exp_var_r = pca.explained_variance_ratio_
        #PC_number = np.arange(pca.n_components_) + 1
        #Biplot Data
        pc1 = pc[:,int(self.entry_PC1.get())-1]
        pc2 = pc[:,int(self.entry_PC2.get())-1]
        loadings = pca.components_
        scalePC1 = 1.0/(pc1.max()-pc1.min())
        scalePC2 = 1.0/(pc2.max()-pc2.min())
        self.save_pcs(pc1,pc2)
        features = list(range(1,self.vectors_per_layer[self.layer].shape[1]+1))
        self.biplot(loadings, features, exp_var_r, pc1, pc2, scalePC1, scalePC2)


    def biplot(self, loadings, features, exp_var_r, PC1, PC2, scalePC1, scalePC2):
        fig = Figure(figsize=(8, 8), dpi=100)
        ax = fig.add_subplot(111)
        for i, feature in enumerate(features):
            ax.arrow(0,0,loadings[0,i],loadings[1,i], head_width = 0.01, head_length = 0.01, color='black', linewidth=1.3)
            ax.text(loadings[0,i]*1.1, loadings[1,i]*1.1, feature, fontsize = 8, color='black')
            ax.scatter(PC1*scalePC1, PC2*scalePC2, s =11)
        ax.set_xlabel(f'PC{int(self.entry_PC1.get())} ({exp_var_r[int(self.entry_PC1.get())-1]*100:.2f}%)', fontsize=12)
        ax.set_ylabel(f'PC{int(self.entry_PC2.get())} ({exp_var_r[int(self.entry_PC2.get())-1]*100:.2f}%)', fontsize=12)
        ax.set_title('Biplot', fontsize=13)

        canvas = FigureCanvasTkAgg(fig, master=self.set2)
        canvas_widget = canvas.get_tk_widget()
        # Usar el sistema de cuadrícula para colocar el lienzo en la ventana
        canvas_widget.grid(row=0, column=3, padx=10, pady=10, rowspan=25, columnspan=20)
        canvas_widget.config(width=900, height=640)      


    def score_clustering_view(self):

        #select an clustering algorithm 
        label_sel_clus_alg = tk.Label(self.set3, text="Selecciona un algoritmo de clustering: ", font="Arial 14")
        label_sel_clus_alg.grid(row=1, column=1, padx=15, pady=15)

        # Variable para almacenar la opción seleccionada
        self.radio_clus_var = tk.StringVar(self.set3, "-1")
        radioClus1 = tk.Radiobutton(self.set3, text="K-means", variable=self.radio_clus_var, value=1, command=self.instances['go_to_k_means'])
        radioClus2 = tk.Radiobutton(self.set3, text="DBSCAN", variable=self.radio_clus_var, value=2, command=self.instances['go_to_DBSCAN'])
        radioClus3 = tk.Radiobutton(self.set3, text="Mean-shift", variable=self.radio_clus_var, value=3, command=self.instances['go_to_mean_shift'])
        radioClus4 = tk.Radiobutton(self.set3, text="Aglomerativo", variable=self.radio_clus_var, value=4, command=self.instances['go_to_agglomerative'])
        radioClus5 = tk.Radiobutton(self.set3, text="Spectral", variable=self.radio_clus_var, value=5, command=self.instances['go_to_spectral'])
        radioClus6 = tk.Radiobutton(self.set3, text="Mezcla Gausiana", variable=self.radio_clus_var, value=6, command=self.instances['go_to_gaussian_mixture'])

        # Colocar los radio buttons en la self.set3
        radioClus1.grid(row=3, column=1, padx=15, pady=15)
        radioClus2.grid(row=5, column=1, padx=15, pady=15)
        radioClus3.grid(row=7, column=1, padx=15, pady=15)
        radioClus4.grid(row=9, column=1, padx=15, pady=15)
        radioClus5.grid(row=11, column=1, padx=15, pady=15)
        radioClus6.grid(row=13, column=1, padx=15, pady=15)