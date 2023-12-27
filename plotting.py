import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize


def plot_clustering(instances, X, clusters, algorithm, labels, text):
    instances['window'].update()
    instances['plot'].clear()
    clusters_list = sorted(list(set(clusters)))
    legends = [str(i) if i != -1 else 'Valores atípicos' for i in clusters_list]
    clusters_ = [i if i != -1 else max(clusters_list)+1 for i in clusters]
    clusters_list = [i if i != -1 else max(clusters_list)+1 for i in clusters_list]
    #n_clusters = int(clusters.n_clusters)
    n_clusters = len(np.unique(clusters))
    # Obtén una paleta de colores para asignar a los clusters
    cmap = cm.get_cmap('tab20', n_clusters)   
    # Grafica los puntos y asigna colores diferentes a cada cluster
    for i, cluster in enumerate(clusters_list):
        cluster_points = X[clusters_ == cluster]
        instances['plot'].scatter(cluster_points[:, 0], cluster_points[:, 1], color=cmap(cluster), label=f'Cluster {legends[i]}')#s = 12
        if text == 'cluster':
          for x, y in cluster_points:
            instances['plot'].text(x, y, str(cluster), color='black', fontsize=8, ha='center', va='center')

    if text == 'label':
      for i, (x, y) in enumerate(X):
          instances['plot'].text(x, y, str(labels[i]), color='black', fontsize=8, ha='center', va='center')
    elif text == 'index':
      for i, (x, y) in enumerate(X):
          instances['plot'].text(x, y, str(i), color='black', fontsize=8, ha='center', va='center')

    instances['original_xlim'] = instances['plot'].get_xlim()
    instances['original_ylim'] = instances['plot'].get_ylim()
    instances['canvas'].draw()
    instances['plot'].set_title(algorithm + ' Clustering')
    instances['plot'].legend()

    #canvas = FigureCanvasTkAgg(fig, master=instances['k_means_window'])
    instances['canvas_widget'].place(x=180, y=-60, width=1200, height=850)
    ##plt.show()



def highlight_linguistic_plot(instances, X, clusters, labels, text, title, legends, words_size, n = 0, restrict_labels=True):
    instances['window'].update()
    instances['plot'].clear()
    clusters_list = sorted(list(set(clusters)))
    n_clusters = len(np.unique(clusters))
    cmap = cm.get_cmap('tab20', n_clusters)   #paleta de colores
    norm = Normalize(vmin=0, vmax=n_clusters - 1)
    #legends_ = [f'Cluster {i}' if isinstance(i, (int)) else f'{i}' for i in legends]
    legends_ = [f'Cluster {i}' if isinstance(i, (int)) else f'{i}' for i in legends]
    # Grafica los puntos y asigna colores diferentes
    #clusters_without_labels = n_clusters - words_size - n if "Valores atípicos" in legends else n_clusters - words_size
    clusters_without_labels = n_clusters - words_size - n

    cp = X[clusters > clusters_without_labels] 
    for i, cluster in enumerate(clusters_list):
        cluster_points = X[clusters == cluster]
        instances['plot'].scatter(cluster_points[:, 0], cluster_points[:, 1], c=[cmap(norm(i))], label=f'{legends_[i]}')
        if text == 'cluster' and not restrict_labels:
          for x, y in cluster_points:
            instances['plot'].text(x, y, str(cluster), color='black', fontsize=8, ha='center', va='center')
        if text == 'cluster' and restrict_labels:
          for x, y in cp:
            instances['plot'].text(x, y, str(cluster), color='black', fontsize=8, ha='center', va='center')

    if text == 'label' and not restrict_labels:
      for i, (x, y) in enumerate(X):
          instances['plot'].text(x, y, str(labels[i]), color='black', fontsize=8, ha='center', va='center')
    elif text == 'index' and not restrict_labels:
      for i, (x, y) in enumerate(X):
          instances['plot'].text(x, y, str(i), color='black', fontsize=8, ha='center', va='center')
    if text == 'label' and restrict_labels:
      for i, (x, y) in enumerate(X):
          if clusters[i] > clusters_without_labels:
            instances['plot'].text(x, y, str(labels[i]), color='black', fontsize=8, ha='center', va='center')
    elif text == 'index' and restrict_labels:
      for i, (x, y) in enumerate(X):
          if clusters[i] > clusters_without_labels:
            instances['plot'].text(x, y, str(i), color='black', fontsize=8, ha='center', va='center')

    instances['original_xlim'] = instances['plot'].get_xlim()
    instances['original_ylim'] = instances['plot'].get_ylim()
    instances['canvas'].draw()
    instances['plot'].set_title(title, fontsize=16)
    # Agregar leyenda
    instances['plot'].legend(fontsize=15)#fontsize=15
    instances['canvas_widget'].grid(row=0, column=6, rowspan=20, padx=10, pady=5, columnspan=5)
    instances['canvas_widget'].config(width=800, height=540)
    instances['plot'].figure.savefig('dis', dpi=500, bbox_inches='tight', pad_inches=0)




# Función para implementar el zoom en un punto específico
def zoom_point(event, zoom_factor, instances):
    x, y = event.x, event.y  # Obtiene las coordenadas del evento de clic
    x_data, y_data = instances['plot'].transData.inverted().transform([x, y])  # Convierte las coordenadas de píxeles a coordenadas de datos
    current_xlim = instances['plot'].get_xlim()
    current_ylim = instances['plot'].get_ylim()

    new_xlim = [
        x_data - (x_data - current_xlim[0]) * zoom_factor,
        x_data + (current_xlim[1] - x_data) * zoom_factor
    ]
    
    new_ylim = [
        y_data - (y_data - current_ylim[0]) * zoom_factor,
        y_data + (current_ylim[1] - y_data) * zoom_factor
    ]

    instances['plot'].set_xlim(new_xlim)
    instances['plot'].set_ylim(new_ylim)
    # Recalcular y ajustar las coordenadas de las etiquetas
    instances['canvas'].draw()

# Función para cambiar el factor de zoom entre zoom in y zoom out
def toggle_zoom_direction(instances):
    if instances['zoom_factor'][0] == 1.2:
        instances['zoom_factor'][0] = 0.8
        instances['zoom_direction_button'].config(text="Zoom In")
    else:
        instances['zoom_factor'][0] = 1.2
        instances['zoom_direction_button'].config(text="Zoom Out")


# Función para restaurar la escala original
def reset_scale(instances):
    instances['plot'].set_xlim(instances['original_xlim'])
    instances['plot'].set_ylim(instances['original_ylim'])
    # Restaurar las coordenadas de las etiquetas originales
    instances['canvas'].draw()

def text_options(instances):
    point_label = instances['radio_var_label_plot'].get()
    if point_label  == "2":
        return 'index'
    elif point_label == "3":
        return 'label'
    elif point_label == "4":
        return 'cluster'
    return ''