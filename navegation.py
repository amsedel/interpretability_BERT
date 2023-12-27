import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import os
from constants import *
import warnings
from autoencoderLSTM_logic import *
from clustering_alg import *
#from k_means import *
from plot_interface import Plot_Interface
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from metrics_interface import Metrics_Interface
from kmeans import K_means
from dbscan import Dbscan
from meanshift import Mean_Shift
from agglomerative import Agglomerative
from spectral import Spectral
from gaussian_mixture import Gaussian_Mixture
from sweeping_K import Sweeping_K
from box_plots import Box_Plot
from linguistic_analysis import Linguistic_Analysis
from pca import PCA_

warnings.filterwarnings("ignore", category=UserWarning)


# Función para cambiar a la segunda interfaz
def go_to_compression_window():
    # Ocultar la ventana actual
    init.withdraw()
    AE_window.withdraw()
    PCA_window.withdraw()
    clustering_window.withdraw()
    metrics_window.withdraw()
    one_cluster_window.withdraw()
    # Mostrar la segunda ventana
    compression_window.deiconify()

# Función para cambiar a la segunda interfaz
def go_to_PCA():
    # Ocultar la ventana actual
    init.withdraw()
    compression_window.withdraw()
    AE_window.withdraw()
    clustering_window.withdraw()
    one_cluster_window.withdraw()
    # Mostrar la segunda ventana
    PCA_window.deiconify()

# Función para cambiar a la segunda interfaz
def go_to_AE():
    # Ocultar la ventana actual
    init.withdraw()
    compression_window.withdraw()
    clustering_window.withdraw()
    metrics_window.withdraw()
    one_cluster_window.withdraw()
    # Mostrar la segunda ventana
    AE_window.deiconify()


# Función para cambiar a la segunda interfaz
def go_to_clustering():
    # Ocultar la ventana actual
    init.withdraw()
    compression_window.withdraw()
    AE_window.withdraw()
    metrics_window.withdraw()
    one_cluster_window.withdraw()
    # Mostrar la segunda ventana
    clustering_window.deiconify()

# Función para cambiar a la segunda interfaz
def go_to_k_means():
    # Ocultar la ventana actual
    init.withdraw()
    compression_window.withdraw()
    AE_window.withdraw()
    #clustering_window.withdraw()
    # Mostrar la segunda ventana
    k_means_window.deiconify()

def go_to_calculate_k():
        # Ocultar la ventana actual
    init.withdraw()
    compression_window.withdraw()
    AE_window.withdraw()
    one_cluster_window.withdraw()
    #clustering_window.withdraw()
    # Mostrar la segunda ventana
    #k_means_window.deiconify()
    calculate_k_window.deiconify()

def go_to_DBSCAN():
    # Ocultar la ventana actual
    init.withdraw()
    compression_window.withdraw()
    AE_window.withdraw()
    calculate_k_window.withdraw()
    #clustering_window.withdraw()
    # Mostrar la segunda ventana
    DBSCAN_window.deiconify()

def go_to_calculate_DBSCAN():
    # Ocultar la ventana actual
    init.withdraw()
    compression_window.withdraw()
    AE_window.withdraw()
    one_cluster_window.withdraw()
    #clustering_window.withdraw()
    # Mostrar la segunda ventana
    #k_means_window.deiconify()
    calculate_dbscan_window.deiconify()

def go_to_mean_shift():
    # Ocultar la ventana actual
    init.withdraw()
    compression_window.withdraw()
    AE_window.withdraw()
    one_cluster_window.withdraw()
    #clustering_window.withdraw()
    # Mostrar la segunda ventana
    mean_shift_window.deiconify()

def go_to_calculate_bandwidth():
    # Ocultar la ventana actual
    init.withdraw()
    compression_window.withdraw()
    AE_window.withdraw()
    one_cluster_window.withdraw()
    #clustering_window.withdraw()
    # Mostrar la segunda ventana
    calculate_meanshift_window.deiconify()

def go_to_agglomerative():
    init.withdraw()
    compression_window.withdraw()
    AE_window.withdraw()
    one_cluster_window.withdraw()
    #clustering_window.withdraw()
    # Mostrar la segunda ventana
    agglomerative_window.deiconify()


def go_to_calculate_kagglomerative():
    # Ocultar la ventana actual
    init.withdraw()
    compression_window.withdraw()
    AE_window.withdraw()
    one_cluster_window.withdraw()
    #clustering_window.withdraw()
    # Mostrar la segunda ventana
    calculate_agglomerative_window.deiconify()


def go_to_spectral():
    init.withdraw()
    compression_window.withdraw()
    AE_window.withdraw()
    one_cluster_window.withdraw()
    #clustering_window.withdraw()
    # Mostrar la segunda ventana
    spectral_window.deiconify()


def go_to_calculate_spectral():
    # Ocultar la ventana actual
    init.withdraw()
    compression_window.withdraw()
    AE_window.withdraw()
    one_cluster_window.withdraw()
    #clustering_window.withdraw()
    # Mostrar la segunda ventana
    calculate_spectral_window.deiconify()


def go_to_gaussian_mixture():
    init.withdraw()
    compression_window.withdraw()
    AE_window.withdraw()
    one_cluster_window.withdraw()
    #clustering_window.withdraw()
    # Mostrar la segunda ventana
    gaussian_mixture_window.deiconify()


def go_to_calculate_gaussian_mixture():
    # Ocultar la ventana actual
    init.withdraw()
    compression_window.withdraw()
    AE_window.withdraw()
    one_cluster_window.withdraw()
    #clustering_window.withdraw()
    # Mostrar la segunda ventana
    calculate_gaussian_mixture_window.deiconify()


def go_to_metrics():
    init.withdraw()
    compression_window.withdraw()
    AE_window.withdraw()
    one_cluster_window.withdraw()
    #clustering_window.withdraw()
    # Mostrar la segunda ventana
    metrics_window.deiconify()

def go_to_view_one_cluster():
    init.withdraw()
    compression_window.withdraw()
    AE_window.withdraw()
    calculate_k_window.withdraw()
    #clustering_window.withdraw()
    # Mostrar la segunda ventana
    one_cluster_window.deiconify()

def go_to_linguistic_analysis():
    init.withdraw()
    compression_window.withdraw()
    AE_window.withdraw()
    one_cluster_window.withdraw()
    #clustering_window.withdraw()
    # Mostrar la segunda ventana
    linguistic_analysis_window.deiconify()

def go_to_sweeping_k():
    init.withdraw()
    compression_window.withdraw()
    AE_window.withdraw()
    one_cluster_window.withdraw()
    #clustering_window.withdraw()
    # Mostrar la segunda ventana
    sweeping_k_window.deiconify()

def go_to_boxs_plots():
    init.withdraw()
    compression_window.withdraw()
    AE_window.withdraw()
    one_cluster_window.withdraw()
    #clustering_window.withdraw()
    # Mostrar la segunda ventana
    box_plots_window.deiconify()


def go_to_view_all_plots():
    init.withdraw()
    compression_window.withdraw()
    AE_window.withdraw()
    one_cluster_window.withdraw()
    #clustering_window.withdraw()
    # Mostrar la segunda ventana
    view_all_plots_window.deiconify()


# Función para volver a la primera interfaz
def go_to_init():
    # Ocultar la ventana actual
    compression_window.withdraw()
    # Mostrar la primera ventana
    init.deiconify()




# Crear ventana de inicio
init = tk.Tk()
init.geometry(str(TK_INIT_WINDOW_X)+"x"+str(TK_INIT_WINDOW_Y))
init.configure(bg="black")

# Crear un botón en la primera ventana
init_btn = tk.Button(init,  text="Inicio", width=10, height=4, font = ("Arial", 28), command=go_to_compression_window)
init_btn.pack(expand=True, fill="both") 
#init_btn.place(relx=0.5, rely=0.5, anchor="center")






# Crear ventana de compresión
compression_window = tk.Tk()
compression_window.title("Compresión de datos")
compression_window.geometry(str(TK_COMPRESSION_WINDOW_X)+"x"+str(TK_COMPRESSION_WINDOW_Y))
compression_window.withdraw()  # Ocultar la segunda ventana al inicio

# Crear un botón en la segunda ventana para volver a inicio
btn_compression_back = tk.Button(compression_window, text="Inicio", command=go_to_init)
btn_compression_back.place(relx=0.1, rely=0.9, anchor=tk.CENTER)


radio_embedding_type_analysis = tk.StringVar(compression_window, "Attention")
radiom_emb1_clus = tk.Radiobutton(compression_window, text="Autoatenciones", variable=radio_embedding_type_analysis, value='Attention')
radiom_emb2_clus = tk.Radiobutton(compression_window, text="Token CLS", variable=radio_embedding_type_analysis, value='CLS')
radiom_emb1_clus.place(relx=0.38, rely=0.12)
radiom_emb2_clus.place(relx=0.38, rely=0.22)

#Mostrar mensaje
label_compression = tk.Label(compression_window, text="Selecciona un método de reducción: ", font="Arial 16")
label_compression.place(relx=0.5, rely=0.42, anchor=tk.CENTER)

# Crear radio buttons
# Variable para almacenar la opción seleccionada
radio_var = tk.StringVar(compression_window, "-1")
radio1 = tk.Radiobutton(compression_window, text="PCA", variable=radio_var, value=1, command=go_to_PCA)
radio2 = tk.Radiobutton(compression_window, text="Autoencoder", variable=radio_var, value=2, command=go_to_AE)
radio3 = tk.Radiobutton(compression_window, text="Cargar vectores reducidos", variable=radio_var, value=3, command=go_to_clustering)

# Colocar los radio buttons en la compression_window
radio1.place(relx=0.3, rely=0.55)
radio2.place(relx=0.3, rely=0.65)
radio3.place(relx=0.3, rely=0.75)




# Crear ventana de compresión por PCA
PCA_window = tk.Tk()
PCA_window.title("PCA")
PCA_window.geometry(str(TK_PCA_WINDOW_X)+"x"+str(TK_PCA_WINDOW_Y))
compression_window.withdraw()
PCA_window.withdraw()


MODEL_CONFIGURATION = {
    'model': 'BERT Base',
    'layers': 12,
    'heads': 12,
    }

instances_pca = {
    'window': PCA_window,
    'config': MODEL_CONFIGURATION,
    #'entry_layer': entry_layer,
    'go_to_compression_window': go_to_compression_window,
    'go_to_k_means': go_to_k_means,
    'go_to_DBSCAN': go_to_DBSCAN,
    'go_to_mean_shift': go_to_mean_shift,
    'go_to_agglomerative': go_to_agglomerative,
    'go_to_spectral': go_to_spectral,
    'go_to_gaussian_mixture': go_to_gaussian_mixture,
    'go_to_linguistic_analysis': go_to_linguistic_analysis,
    'radio_embedding_type_analysis': radio_embedding_type_analysis,
    'radio_var': radio_var
}


pca_ = PCA_(instances_pca)
pca_.pca_view()





# Crear ventana de compresión por Autoencoder LSTM
AE_window = tk.Tk()
AE_window.title("Compresión de datos por Autoencoder")
AE_window.geometry(str(TK_AELSTM_WINDOW_X)+"x"+str(TK_AELSTM_WINDOW_Y))
compression_window.withdraw()
AE_window.withdraw()


btn_AE_LSTM_back = tk.Button(AE_window, text="<-", command=go_to_compression_window)
btn_AE_LSTM_back.place(relx=0.1, rely=0.9, anchor=tk.CENTER)

#Mostrar mensaje
label_model = tk.Label(AE_window, text="Modelo: ", font="Arial 14")
label_model.place(relx=0.11, rely=0.085, anchor=tk.CENTER)

radio_model = tk.StringVar(AE_window, "1")
radiom = tk.Radiobutton(AE_window, text="BERT Base", variable=radio_model, value=1)
radiom.place(relx=0.2, rely=0.06)


label_load_pth = tk.Label(AE_window, text="Introduce la ruta o nombre del objeto .pth: ", font="Arial 14")
label_load_pth.place(relx=0.28, rely=0.14, anchor=tk.CENTER)
entry_root = tk.Entry(AE_window)
entry_root.place(relx=0.67, rely=0.14, anchor=tk.CENTER)
btn_load_pth = tk.Button(AE_window, text="Cargar .pth", command=lambda: load_pth(instances_AE, tk, messagebox, MODEL_CONFIGURATION))
btn_load_pth.place(relx=0.15, rely=0.2, anchor=tk.CENTER)
label_valid = tk.Label(AE_window, text=" ", font="Arial 12", fg="GRAY")
label_valid.place(relx=0.35, rely=0.2, anchor=tk.CENTER)
label_elements = tk.Label(AE_window, text=" ", font="Arial 14")
label_elements.place(relx=0.675, rely=0.21, anchor=tk.CENTER)

label_epochs = tk.Label(AE_window, text="Número de épocas: ", font="Arial 12")
label_epochs.place_forget()
entry_epochs_default = tk.StringVar(AE_window, "5")
entry_epochs = tk.Entry(AE_window, textvariable=entry_epochs_default)
entry_epochs.place_forget()

label_lr = tk.Label(AE_window, text="Tasa de aprendizaje: ", font="Arial 12")
label_lr.place_forget()
entry_lr_default = tk.StringVar(AE_window, "0.001")

entry_lr = tk.Entry(AE_window, textvariable=entry_lr_default)
entry_lr.place_forget()

label_optim = tk.Label(AE_window, text="Optimizador: ", font="Arial 12")
label_optim.place_forget()
radio_optim = tk.StringVar(AE_window, "1")
radiom = tk.Radiobutton(AE_window, text="Adam", variable=radio_optim, value=1)
radiom.place_forget()
label_dim_reduction = tk.Label(AE_window, text="Dimensión de salida: ", font="Arial 12")
label_dim_reduction.place_forget()
entry_dim_reduction_default = tk.StringVar(AE_window, "2")
entry_dim_reduction = tk.Entry(AE_window, textvariable=entry_dim_reduction_default)
entry_dim_reduction.place_forget()

btn_train = tk.Button(AE_window, text="Entrenar", command=lambda: train_AE(instances_AE, MODEL_CONFIGURATION))
btn_train.place_forget()

# Show training process
frame_training = tk.Frame(AE_window, width=480, height=80)  # Establece el ancho y alto del frame
#frame.place(x=50, y=0)  # Posición del frame
frame_training.place_forget()
canvas_training = tk.Canvas(frame_training, width=480, height=80)
canvas_training.pack_forget()
scrollbar_training = tk.Scrollbar(frame_training, command=canvas_training.yview)
scrollbar_training.pack_forget()
canvas_training.configure(yscrollcommand=scrollbar_training.set)
text_widget = tk.Text(canvas_training, wrap='none')  # 'none' significa que no se envolverá automáticamente
canvas_training.create_window((0, 0), window=text_widget, anchor='nw')

#save embedded
dir_name = os.path.dirname(os.path.abspath(__file__)) + '/'
entry_save_reduction_default = tk.StringVar(AE_window, "reduced_vectors.pth")
entry_save_reduction = tk.Entry(AE_window, textvariable=entry_save_reduction_default)
entry_save_reduction.place_forget()
btn_save_emb_red = tk.Button(AE_window, text="Guardar vectores reducidos", command=lambda: save_embedded_reduced(instances_AE, MODEL_CONFIGURATION, messagebox))
btn_save_emb_red.place_forget()

label_save_pth = tk.Label(AE_window, text=" ", font="Arial 12", fg="GRAY")
label_save_pth.place(relx=0.13, rely=0.79, anchor=tk.CENTER)

btn_AE_to_clustering = tk.Button(AE_window, text="Clustering ->", command=go_to_clustering)
btn_AE_to_clustering.place_forget()


if radio_model.get() == "1":
    MODEL_CONFIGURATION = {
        'model': 'BERT Base',
        'layers': 12,
        'heads': 12,
        'batch_size': 12,
        'root': dir_name
    }

instances_AE = {
    'entry_root': entry_root,
    'label_valid': label_valid,
    'label_elements': label_elements,
    'label_epochs' : label_epochs,
    'entry_epochs' : entry_epochs,
    'label_lr' : label_lr,
    'entry_lr' : entry_lr,
    'label_optim': label_optim,
    'radiom': radiom,
    'label_dim_reduction': label_dim_reduction,
    'entry_dim_reduction': entry_dim_reduction,
    'btn_train': btn_train,
    'frame_training': frame_training,
    'canvas_training': canvas_training,
    'scrollbar_training': scrollbar_training,
    'text_widget': text_widget,
    'entry_save_reduction': entry_save_reduction,
    'btn_save_emb_red': btn_save_emb_red,
    'label_save_pth': label_save_pth,
    'btn_AE_to_clustering': btn_AE_to_clustering,
    'embedding_type': radio_embedding_type_analysis,
    'tk': tk
}



# Crear ventana para clustering
clustering_window = tk.Tk()
clustering_window.title("Seleccionar Clustering")
clustering_window.geometry(str(TK_CLUSTERING_WINDOW_X)+"x"+str(TK_CLUSTERING_WINDOW_Y))
compression_window.withdraw()
clustering_window.withdraw()

btn_clustering_back = tk.Button(clustering_window, text="<-", command=go_to_compression_window)
btn_clustering_back.place(relx=0.1, rely=0.9, anchor=tk.CENTER)

label_load_pth_clus = tk.Label(clustering_window, text="Introduce la ruta del objeto .pth de vectores 2D: ", font="Arial 14")
label_load_pth_clus.place(relx=0.4, rely=0.08, anchor=tk.CENTER)
entry_root_reduced_vec = tk.Entry(clustering_window)
entry_root_reduced_vec.place(relx=0.285, rely=0.15, anchor=tk.CENTER)
btn_load_pth_reduced_vec = tk.Button(clustering_window, text="Cargar .pth", command=lambda: load_pth_reduced_vector(instances_CLUS, messagebox, MODEL_CONFIGURATION))
btn_load_pth_reduced_vec.place(relx=0.6, rely=0.145, anchor=tk.CENTER)
label_load_reduced_vec = tk.Label(clustering_window, text=" ", font="Arial 12", fg="GRAY")
label_load_reduced_vec.place(relx=0.81, rely=0.15, anchor=tk.CENTER)

#select an clustering algorithm 
label_sel_clus_analysis = tk.Label(clustering_window, text="Tipo de análisis: ", font="Arial 14")
label_sel_clus_analysis.place_forget()

radio_clus_ana_var = tk.StringVar(clustering_window, "1")
radioClusAna1 = tk.Radiobutton(clustering_window, text="Todas las secuencias por capa", variable=radio_clus_ana_var, value=1)
radioClusAna2 = tk.Radiobutton(clustering_window, text="Todas las capas de una secuencia", variable=radio_clus_ana_var, value=2)
radioClusAna1.place_forget()
radioClusAna2.place_forget()

#select an clustering algorithm 
label_sel_clus_alg = tk.Label(clustering_window, text="Selecciona un algoritmo de clustering: ", font="Arial 14")
label_sel_clus_alg.place_forget()

# Variable para almacenar la opción seleccionada
radio_clus_var = tk.StringVar(clustering_window, "-1")
radioClus1 = tk.Radiobutton(clustering_window, text="K-means", variable=radio_clus_var, value=1, command=go_to_k_means)
radioClus2 = tk.Radiobutton(clustering_window, text="DBSCAN", variable=radio_clus_var, value=2, command=go_to_DBSCAN)
radioClus3 = tk.Radiobutton(clustering_window, text="Mean-shift", variable=radio_clus_var, value=3, command=go_to_mean_shift)
radioClus4 = tk.Radiobutton(clustering_window, text="Aglomerativo", variable=radio_clus_var, value=4, command=go_to_agglomerative)
radioClus5 = tk.Radiobutton(clustering_window, text="Spectral", variable=radio_clus_var, value=5, command=go_to_spectral)
radioClus6 = tk.Radiobutton(clustering_window, text="Mezcla Gausiana", variable=radio_clus_var, value=6, command=go_to_gaussian_mixture)

# Colocar los radio buttons en la clustering_window
radioClus1.place_forget()
radioClus2.place_forget()
radioClus3.place_forget()
radioClus4.place_forget()
radioClus5.place_forget()
radioClus6.place_forget()


label_sel_clus_layer = tk.Label(clustering_window, text="Selecciona la capa BERT (0-11): ", font="Arial 14")
label_sel_clus_layer.place_forget()

entry_layer_default = tk.IntVar(clustering_window, "0")
entry_layer = tk.Entry(clustering_window, width=10, textvariable=entry_layer_default)
entry_layer.place_forget()


instances_CLUS = {
    'entry_root_reduced_vec' : entry_root_reduced_vec,
    'radio_embedding_type_analysis': radio_embedding_type_analysis,
    'label_load_reduced_vec' : label_load_reduced_vec,
    'label_sel_clus_analysis': label_sel_clus_analysis,
    'label_sel_clus_alg': label_sel_clus_alg,
    'radioClusAna1': radioClusAna1,
    'radioClusAna2': radioClusAna2,
    'radioClus1': radioClus1,
    'radioClus2': radioClus2,
    'radioClus3': radioClus3,
    'radioClus4': radioClus4,
    'radioClus5': radioClus5,
    'radioClus6': radioClus6,
    'label_sel_clus_layer': label_sel_clus_layer,
    'entry_layer_default': entry_layer_default,
    'entry_layer': entry_layer
}


# Crear ventana de k-means
k_means_window = tk.Tk()
k_means_window.title("Algoritmo de distancia K-means")
k_means_window.geometry(str(TK_K_MEANS_WINDOW_X)+"x"+str(TK_K_MEANS_WINDOW_Y))
k_means_window.withdraw()  # Ocultar la segunda ventana al inicio
#k_means_window.bind("<Enter>", lambda event: get_type_analysis(event, instances_k_means))
instances_k_means = {
    'window': k_means_window,
    'entry_layer': entry_layer,
    'radio_clus_ana_var': radio_clus_ana_var,
    'go_to_calculate_k': go_to_calculate_k,
    'go_to_metrics': go_to_metrics,
    'go_to_view_one_cluster': go_to_view_one_cluster,
    'go_to_linguistic_analysis': go_to_linguistic_analysis,
    'radio_embedding_type_analysis': radio_embedding_type_analysis,
    'radio_var': radio_var,
    'pca': pca_
}


k_means_ = K_means(instances_k_means)
k_means_.view_kmeans()


"""
label_layer = tk.Label(k_means_window, text="Escoge una capa: ", font="Arial 14")
label_layer.place(relx=0.07, rely=0.04, anchor=tk.CENTER)

clus_layer = tk.IntVar(k_means_window, 0)
entry_clus_layer = tk.Entry(k_means_window,width=8, textvariable=clus_layer)
entry_clus_layer.place(relx=0.065, rely=0.07, anchor=tk.CENTER)

label_k = tk.Label(k_means_window, text="Define K: ", font="Arial 14")
label_k.place(relx=0.07, rely=0.11, anchor=tk.CENTER)
entry_k = tk.Entry(k_means_window, width=8)
entry_k.place(relx=0.065, rely=0.14, anchor=tk.CENTER)

btn_k = tk.Button(k_means_window, text="Sugerir K", command=go_to_calculate_k)
btn_k.place(relx=0.065, rely=0.20, anchor=tk.CENTER)

instances_k_means = {
    'window': k_means_window,
    'radio_clus_ana_var': radio_clus_ana_var,
    'entry_clus_layer': entry_clus_layer,
    'entry_k': entry_k,
    'messagebox': messagebox,
    'go_to_metrics': go_to_metrics,
    'go_to_view_one_cluster': go_to_view_one_cluster,
    'tk': tk
}

plot_k_means = Plot_Interface(instances_k_means,[1.2],(6,6))
plot_k_means.view_btn("K-means", k_means)
plot_k_means.view_radios_btn()
plot_k_means.view_plot()

instances_k_means['plot'] = plot_k_means.plot
instances_k_means['canvas'] = plot_k_means.canvas
instances_k_means['canvas_widget'] =plot_k_means.canvas_widget
instances_k_means['zoom_factor'] =plot_k_means.zoom_factor
instances_k_means['zoom_direction_button'] = plot_k_means.btn_zoom_direction
instances_k_means['radio_var_label_plot'] = plot_k_means.radio_var_label_plot

"""

# Crear ventana de Elbow
calculate_k_window = tk.Tk()
calculate_k_window.title("Métodos de Elbow y silhouette")
calculate_k_window.geometry(str(TK_CALCULATE_K_WINDOW_X)+"x"+str(TK_CALCULATE_K_WINDOW_Y))
calculate_k_window.withdraw()  # Ocultar la segunda ventana al inicio
"""
label_k_elbow = tk.Label(calculate_k_window, text="Valor de K hasta donde deseas probar: ", font="Arial 14")
label_k_elbow.place(relx=0.26, rely=0.075, anchor=tk.CENTER)

clus_k_elbow = tk.IntVar(calculate_k_window, 10)
entry_k_elbow  = tk.Entry(calculate_k_window, textvariable=clus_k_elbow)
entry_k_elbow.place(relx=0.64, rely=0.075, anchor=tk.CENTER)
btn_k_elbow = tk.Button(calculate_k_window, text="Calcular K", command=lambda: calculate_k(instances_calculate_k))
btn_k_elbow.place(relx=0.89, rely=0.074, anchor=tk.CENTER)

best_k_sil = tk.Label(calculate_k_window, text=" ", font="Arial 16")
best_k_sil.place(relx=0.5, rely=0.15, anchor=tk.CENTER)

instances_calculate_k = {
    'entry_clus_layer': entry_clus_layer,
    'window': calculate_k_window,
    'entry_k_elbow': entry_k_elbow,
    'best_k_sil': best_k_sil,
    'messagebox': messagebox,
    'tk': tk
}
"""
instances_k_means['calculate_k_window'] = calculate_k_window
k_means_.view_calculate_k()



# Crear ventana de DBSCAN
DBSCAN_window = tk.Tk()
DBSCAN_window.title("Algoritmo de densidad DBSCAN")
DBSCAN_window.geometry(str(TK_DBSCAN_WINDOW_X)+"x"+str(TK_DBSCAN_WINDOW_Y))
DBSCAN_window.withdraw()  # Ocultar la segunda ventana al inicio


instances_dbscan = {
    'window': DBSCAN_window,
    'entry_layer': entry_layer,
    'radio_clus_ana_var': radio_clus_ana_var,
    'go_to_calculate_eps_n_min': go_to_calculate_DBSCAN,
    'go_to_metrics': go_to_metrics,
    'go_to_view_one_cluster': go_to_view_one_cluster,
    'go_to_linguistic_analysis': go_to_linguistic_analysis,
    'radio_embedding_type_analysis': radio_embedding_type_analysis,
    'radio_var': radio_var,
    'pca': pca_
}


dbscan_ = Dbscan(instances_dbscan)
dbscan_.view_dbscan()


# Crear ventana de calculo de parámetros de DBSCAN
calculate_dbscan_window = tk.Tk()
calculate_dbscan_window.title("Cálculo de parámetros DBSCAN")
calculate_dbscan_window.geometry(str(TK_CALCULATE_DBSCAN_WINDOW_X)+"x"+str(TK_CALCULATE_DBSCAN_WINDOW_Y))
calculate_dbscan_window.withdraw()  # Ocultar la segunda ventana al inicio


instances_dbscan['calculate_eps_nmin_window'] = calculate_dbscan_window
dbscan_.view_calculate_dbscan()




# Crear ventana de Mean shift
mean_shift_window = tk.Tk()
mean_shift_window.title("Algoritmo de densidad mean shift")
mean_shift_window.geometry(str(TK_MEAN_SHIFT_WINDOW_X)+"x"+str(TK_MEAN_SHIFT_WINDOW_Y))
mean_shift_window.withdraw()  # Ocultar la segunda ventana al inicio


instances_mean_shift = {
    'window': mean_shift_window,
    'entry_layer': entry_layer,
    'radio_clus_ana_var': radio_clus_ana_var,
    'go_to_calculate_bandwidth': go_to_calculate_bandwidth,
    'go_to_metrics': go_to_metrics,
    'go_to_view_one_cluster': go_to_view_one_cluster,
    'go_to_linguistic_analysis': go_to_linguistic_analysis,
    'radio_embedding_type_analysis': radio_embedding_type_analysis,
    'radio_var': radio_var,
    'pca': pca_
}

mean_shift_ = Mean_Shift(instances_mean_shift)
mean_shift_.view_meanshift()


# Crear ventana de calculo de parámetros de Mean-Shift
calculate_meanshift_window = tk.Tk()
calculate_meanshift_window.title("Cálculo de parámetros Mean-shift")
calculate_meanshift_window.geometry(str(TK_CALCULATE_MEAN_SHIFT_WINDOW_X)+"x"+str(TK_CALCULATE_MEAN_SHIFT_WINDOW_Y))
calculate_meanshift_window.withdraw()  # Ocultar la segunda ventana al inicio

instances_mean_shift['calculate_bandwidth_window'] = calculate_meanshift_window
mean_shift_.view_calculate_bandwidth()





#Crean ventana de algoritmo aglomerativo
agglomerative_window = tk.Tk()
agglomerative_window.title("Algoritmo jerárquico Aglomerativo")
agglomerative_window.geometry(str(TK_AGLOMERATIVE_WINDOW_X)+"x"+str(TK_AGLOMERATIVE_WINDOW_Y))
agglomerative_window.withdraw()  # Ocultar la segunda ventana al inicio

instances_agglomerative = {
    'window': agglomerative_window,
    'entry_layer': entry_layer,
    'radio_clus_ana_var': radio_clus_ana_var,
    'go_to_calculate_kagglomerative': go_to_calculate_kagglomerative,
    'go_to_metrics': go_to_metrics,
    'go_to_view_one_cluster': go_to_view_one_cluster,
    'go_to_linguistic_analysis': go_to_linguistic_analysis,
    'radio_embedding_type_analysis': radio_embedding_type_analysis,
    'radio_var': radio_var,
    'pca': pca_
}

agglomerative_ = Agglomerative(instances_agglomerative)
agglomerative_.view_agglomerative()

# Crear ventana de calculo de parámetros de Agglomerative
calculate_agglomerative_window = tk.Tk()
calculate_agglomerative_window.title("Cálculo de parámetros Aglomerativo")
calculate_agglomerative_window.geometry(str(TK_CALCULATE_AGLOMERATIVE_WINDOW_X)+"x"+str(TK_CALCULATE_AGLOMERATIVE_WINDOW_Y))
calculate_agglomerative_window.withdraw()  # Ocultar la segunda ventana al inicio

instances_agglomerative['calculate_agglomerative_window'] = calculate_agglomerative_window
agglomerative_.view_calculate_kagglomerative()




#Crean ventana de algoritmo spectral
spectral_window = tk.Tk()
spectral_window.title("Algoritmo Spectral")
spectral_window.geometry(str(TK_SPECTRAL_WINDOW_X)+"x"+str(TK_SPECTRAL_WINDOW_Y))
spectral_window.withdraw()  # Ocultar la segunda ventana al inicio

instances_spectral = {
    'window': spectral_window,
    'entry_layer': entry_layer,
    'radio_clus_ana_var': radio_clus_ana_var,
    'go_to_calculate_spectral': go_to_calculate_spectral,
    'go_to_metrics': go_to_metrics,
    'go_to_view_one_cluster': go_to_view_one_cluster,
    'go_to_linguistic_analysis': go_to_linguistic_analysis,
    'radio_embedding_type_analysis': radio_embedding_type_analysis,
    'radio_var': radio_var,
    'pca': pca_
}

spectral_ = Spectral(instances_spectral)
spectral_.view_spectral()

# Crear ventana de calculo de parámetros de Agglomerative
calculate_spectral_window = tk.Tk()
calculate_spectral_window.title("Cálculo de parámetros SpectralClustering")
calculate_spectral_window.geometry(str(TK_CALCULATE_SPECTRAL_WINDOW_X)+"x"+str(TK_CALCULATE_SPECTRAL_WINDOW_Y))
calculate_spectral_window.withdraw()  # Ocultar la segunda ventana al inicio

instances_spectral['calculate_spectral_window'] = calculate_spectral_window
spectral_.view_calculate_spectral()





#Crean ventana de algoritmo gaussian_mixture
gaussian_mixture_window = tk.Tk()
gaussian_mixture_window.title("Algoritmo de Mezcla Gausiana")
gaussian_mixture_window.geometry(str(TK_GAUSSIAN_MIXTURE_WINDOW_X)+"x"+str(TK_GAUSSIAN_MIXTURE_WINDOW_Y))
gaussian_mixture_window.withdraw()  # Ocultar la segunda ventana al inicio

instances_gaussian_mixture = {
    'window': gaussian_mixture_window,
    'entry_layer': entry_layer,
    'radio_clus_ana_var': radio_clus_ana_var,
    'go_to_calculate_gaussian_mixture': go_to_calculate_gaussian_mixture,
    'go_to_metrics': go_to_metrics,
    'go_to_view_one_cluster': go_to_view_one_cluster,
    'go_to_linguistic_analysis': go_to_linguistic_analysis,
    'radio_embedding_type_analysis': radio_embedding_type_analysis,
    'radio_var': radio_var,
    'pca': pca_
}

gaussian_mixture_ = Gaussian_Mixture(instances_gaussian_mixture)
gaussian_mixture_.view_gaussian_mixture()

# Crear ventana de calculo de parámetros de Agglomerative
calculate_gaussian_mixture_window = tk.Tk()
calculate_gaussian_mixture_window.title("Cálculo de parámetros para mezcla gausiana")
calculate_gaussian_mixture_window.geometry(str(TK_CALCULATE_GAUSSIAN_MIXTURE_WINDOW_X)+"x"+str(TK_CALCULATE_GAUSSIAN_MIXTURE_WINDOW_Y))
calculate_gaussian_mixture_window.withdraw()  # Ocultar la segunda ventana al inicio

instances_gaussian_mixture['calculate_gaussian_mixture_window'] = calculate_gaussian_mixture_window
gaussian_mixture_.view_calculate_gaussian_mixture()





metrics_window = tk.Tk()
metrics_window.title("Métricas de Clustering")
metrics_window.geometry(str(TK_METRICS_WINDOW_X)+"x"+str(TK_METRICS_WINDOW_Y))
metrics_window.withdraw()  # Ocultar la segunda ventana al inicio


metrics_instances= {
    'entry_layer': entry_layer,
    'window': metrics_window,
    'radio_clus_ana_var': radio_clus_ana_var,
    'radio_clus_var': radio_clus_var,
    'go_to_sweeping_k': go_to_sweeping_k,
    'kmeans_instance': k_means_,
    'kmeans': k_means_.kmeans,
    'dbscan': dbscan_.dbscan,
    'dbscan_instance':dbscan_,
    'meanshift': mean_shift_.meanshift,
    'meanshift_instance': mean_shift_,
    'agglomerative':agglomerative_.agglomerative,
    'agglomerative_instance': agglomerative_,
    'spectral': spectral_.spectral,
    'spectral_instance': spectral_,
    'gaussian_mixture': gaussian_mixture_.gaussian_mixture,
    'gaussian_mixture_instance': gaussian_mixture_,
    'radio_embedding_type_analysis': radio_embedding_type_analysis,
    'radio_var': radio_var,
    'pca': pca_
}
metrics = Metrics_Interface(metrics_instances)
metrics.view_radio_measure()





sweeping_k_window = tk.Tk()
sweeping_k_window.title("Barrido K")
sweeping_k_window.geometry(str(TK_SWEEPING_K_WINDOW_X)+"x"+str(TK_SWEEPING_K_WINDOW_Y))
sweeping_k_window.withdraw()  # Ocultar la segunda ventana al inicio

sweeping_k_instances= {
    'entry_layer': entry_layer,
    'window': sweeping_k_window,
    'radio_clus_ana_var': radio_clus_ana_var,
    'radio_clus_var': radio_clus_var,
    'kmeans_instance': k_means_,
    'kmeans': k_means_.kmeans,
    'dbscan_instance':dbscan_,
    'dbscan': dbscan_.dbscan,
    'meanshift_instance': mean_shift_,
    'meanshift': mean_shift_.meanshift,
    'agglomerative_instance': agglomerative_,
    'agglomerative':agglomerative_.agglomerative,
    'spectral_instance': spectral_,
    'spectral': spectral_.spectral,
    'gaussian_mixture_instance': gaussian_mixture_,
    'gaussian_mixture': gaussian_mixture_.gaussian_mixture,
    'radio_embedding_type_analysis': radio_embedding_type_analysis,
    'radio_var': radio_var,
    'pca': pca_
}

sweeping_k_ = Sweeping_K(sweeping_k_instances)
sweeping_k_.view_sweeping_k()

def update_sweeping_k(event):
    sweeping_k_.focus_in()

sweeping_k_window.bind("<FocusIn>", update_sweeping_k)




linguistic_analysis_window  = tk.Tk()
linguistic_analysis_window.title("Análisis linguístico")
linguistic_analysis_window.geometry(str(TK_LINGUISTIC_ANALYSIS_X)+"x"+str(TK_LINGUISTIC_ANALYSIS_Y))
linguistic_analysis_window.withdraw()  # Ocultar la segunda ventana al inicio


linguistic_analysis_instances = {
    'entry_layer': entry_layer,
    'window': linguistic_analysis_window,
    'radio_clus_ana_var': radio_clus_ana_var,
    'radio_clus_var': radio_clus_var,
    'kmeans_instance': k_means_,
    'kmeans': k_means_.kmeans,
    'dbscan_instance':dbscan_,
    'dbscan': dbscan_.dbscan,
    'meanshift_instance': mean_shift_,
    'meanshift': mean_shift_.meanshift,
    'agglomerative_instance': agglomerative_,
    'agglomerative': agglomerative_.agglomerative,
    'spectral_instance': spectral_,
    'spectral': spectral_.spectral,
    'gaussian_mixture_instance': gaussian_mixture_,
    'gaussian_mixture': gaussian_mixture_.gaussian_mixture,
    'go_to_boxs_plots': go_to_boxs_plots,
    'radio_embedding_type_analysis': radio_embedding_type_analysis,
    'radio_var': radio_var,
    'pca': pca_
}

linguistic_analysis = Linguistic_Analysis(linguistic_analysis_instances)
linguistic_analysis.analysis_options()
linguistic_analysis.update_view()




#ventana boxplots
box_plots_window = tk.Tk()
box_plots_window.title("Análisis Box plots")
box_plots_window.geometry(str(TK_BOX_PLOTS_WINDOW_X)+"x"+str(TK_BOX_PLOTS_WINDOW_Y))
box_plots_window.withdraw()  # Ocultar la segunda ventana al inicio


view_all_plots_window = tk.Tk()
view_all_plots_window.title("Ver todos los diagramas de caja")
view_all_plots_window.geometry(str(TK_ALL_BOX_PLOTS_WINDOW_X)+"x"+str(TK_ALL_BOX_PLOTS_WINDOW_Y))
view_all_plots_window.withdraw()  # Ocultar la segunda ventana al inicio


box_plot_instances= {
    'entry_layer': entry_layer,
    'window': box_plots_window,
    'radio_clus_ana_var': radio_clus_ana_var,
    'radio_clus_var': radio_clus_var,
    'kmeans_instance': k_means_,
    'kmeans': k_means_.kmeans,
    'dbscan_instance':dbscan_,
    'dbscan': dbscan_.dbscan,
    'meanshift_instance': mean_shift_,
    'meanshift': mean_shift_.meanshift,
    'agglomerative_instance': agglomerative_,
    'agglomerative':agglomerative_.agglomerative,
    'spectral_instance': spectral_,
    'spectral': spectral_.spectral,
    'gaussian_mixture_instance': gaussian_mixture_,
    'gaussian_mixture': gaussian_mixture_.gaussian_mixture,
    'linguistic_analysis': linguistic_analysis,
    'view_all_plots_window': view_all_plots_window,
    'go_to_view_all_plots': go_to_view_all_plots,
    'radio_embedding_type_analysis': radio_embedding_type_analysis,
    'radio_var': radio_var

}

box_plot_ = Box_Plot(box_plot_instances)
box_plot_.view_box_plot()
box_plot_.view_all_box_plot()


one_cluster_window = tk.Tk()
one_cluster_window.title("Métricas de Clustering")
one_cluster_window.geometry(str(TK_ONE_CLUSTER_WINDOW_X)+"x"+str(TK_ONE_CLUSTER_WINDOW_Y))
one_cluster_window.withdraw()  # Ocultar la segunda ventana al inicio