import torch
import numpy as np
import os
import tkinter as tk
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def load_pth_reduced_vector(instances, messagebox, MODEL_CONFIGURATION):
    root = instances['entry_root_reduced_vec'].get()
    type_analysis = instances['radio_embedding_type_analysis'].get()
    root_to_save = os.path.dirname(os.path.abspath(__file__))
    try:
        global reduced_vectors
        reduced_vectors = torch.load(root)
        global vectors_per_layer, sequences, labels
        vectors_per_layer, sequences, labels, dimensions =  all_sequences_per_layer(MODEL_CONFIGURATION)
        instances['label_load_reduced_vec'].config(text=f"Archivo cargado.")
        show_clustering_types(instances)
        data_all_sequences_per_layer = {
           'vectors': vectors_per_layer,
           'sequences': sequences,
           'labels': labels,
           'dimensions': dimensions
        }
        if type_analysis == 'Attention':
          torch.save(data_all_sequences_per_layer, root_to_save + '/data_all_sequences_per_layer_attention.pth')
        if type_analysis == 'CLS':
          torch.save(data_all_sequences_per_layer, root_to_save + '/data_all_sequences_per_layer_CLS.pth')
    except FileNotFoundError:
        # Manejo de la excepción si el archivo no se encuentra
        messagebox.showerror("Error", "El archivo " + root + " no existe.")
    except Exception as e:
        messagebox.showerror("Error", "Se produjo una excepción:" + str(e))


def get_representations_per_layer(num_sentences, vector_representations, config):
  vectors_per_layer = {}
  labels = {}
  for l in range(config['layers']):
    #vectors_per_layer[l] = np.array([vector_representations[(i,l)]['vector'].detach().numpy() for i in range(num_sentences)])
    vectors_per_layer[l] = np.array([vector_representations[(i,l)]['vector'] for i in range(num_sentences)])
  labels = { i: vector_representations[(i,0)]['label'][0].item() for i in range(num_sentences)}
  sequences = { i: vector_representations[(i,0)]['sequence'][0] for i in range(num_sentences)}
  dimensions = { i: vector_representations[(i,0)]['dimension'][0].item() for i in range(num_sentences)}
  return vectors_per_layer, sequences, labels, dimensions


def all_sequences_per_layer(config):
   num_sequences = len(reduced_vectors) // config['layers']
   return get_representations_per_layer(num_sequences, reduced_vectors, config)


def show_clustering_types(instances):
    instances['label_sel_clus_analysis'].place(relx=0.5, rely=0.25, anchor=tk.CENTER)
    instances['radioClusAna1'].place(relx=0.3, rely=0.3)
    instances['radioClusAna2'].place(relx=0.3, rely=0.35)
    instances['label_sel_clus_layer'].place(relx=0.15, rely=0.45)
    instances['entry_layer'].place(relx=0.6, rely=0.45)
    instances['label_sel_clus_alg'].place(relx=0.5, rely=0.58, anchor=tk.CENTER)
    instances['radioClus1'].place(relx=0.4, rely=0.63)
    instances['radioClus2'].place(relx=0.4, rely=0.67)
    instances['radioClus3'].place(relx=0.4, rely=0.71)
    instances['radioClus4'].place(relx=0.4, rely=0.75)
    instances['radioClus5'].place(relx=0.4, rely=0.79)
    instances['radioClus6'].place(relx=0.4, rely=0.83)




