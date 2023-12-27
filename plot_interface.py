
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from plotting import *
from k_means import k_means

class Plot_Interface:
    def __init__(self, instances, zoom_factor = [1.2], plot_size=(6,6), isFrame=False):
        self.instances = instances
        self.isFrame = isFrame
        self.zoom_factor = zoom_factor
        self.plot_size = plot_size
        self.fig = Figure(self.plot_size, dpi=100)
        if self.isFrame:
            self.canvas = FigureCanvasTkAgg(self.fig, master=instances['frame'])
        else:
            self.canvas = FigureCanvasTkAgg(self.fig, master=instances['window'])


    def view_plot(self):
        self.plot = self.fig.add_subplot(111)
        self.plot.set_xlabel('X')
        self.plot.set_ylabel('Y')
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.bind("<Button-1>", lambda event: zoom_point(event, self.zoom_factor[0], self.instances))

        if self.isFrame:
            self.btn_zoom_direction = tk.Button(self.instances['frame'], text="Zoom Out", command=lambda:toggle_zoom_direction(self.instances))
            btn_reset_scale = tk.Button(self.instances['frame'], text="Restaurar Escala", command=lambda:reset_scale(self.instances))
            self.canvas_widget.grid_remove()
            self.btn_zoom_direction.grid(row=0, column=9, sticky="ne")
            btn_reset_scale.grid(row=0, column=10, sticky="ne")
            
        else:
            self.btn_zoom_direction = tk.Button(self.instances['window'], text="Zoom Out", command=lambda:toggle_zoom_direction(self.instances))
            btn_reset_scale = tk.Button(self.instances['window'], text="Restaurar Escala", command=lambda:reset_scale(self.instances))
            self.canvas_widget.place_forget()
            self.btn_zoom_direction.place(relx=0.064, rely=0.9, anchor=tk.CENTER)
            btn_reset_scale.place(relx=0.064, rely=0.95, anchor=tk.CENTER)
            

    def view_btns_options(self, pos = {'bx1':0.064,'by1':0.6,'bx2':0.064, 'by2':0.65, 'bx3': 0.064, 'by3':0.7}):
        btn_metrics = tk.Button(self.instances['window'], text="Métricas", command=self.instances['go_to_metrics'])
        btn_metrics.place(relx=pos['bx1'], rely=pos['by1'], anchor=tk.CENTER)

        btn_analysis_lin = tk.Button(self.instances['window'], text="Análisis Lingüístico", command=self.instances['go_to_linguistic_analysis'])
        btn_analysis_lin.place(relx=pos['bx2'], rely=pos['by2'], anchor=tk.CENTER)

        #btn_one_cluster = tk.Button(self.instances['window'], text="Ver 1 cluster", command=self.instances['go_to_view_one_cluster'])
        #btn_one_cluster.place(relx=pos['bx3'], rely=pos['by3'], anchor=tk.CENTER)

    def view_scale_radios(self, pos = {'rx1':0.065,'ry1':0.04,'rx2':0.065,'ry2':0.07}):
        self.var_norm = tk.StringVar(self.instances['window'], 'no-scale')
        radio1_view = tk.Radiobutton(self.instances['window'], text="No escalado", variable=self.var_norm , value='no-scale')
        radio2_view = tk.Radiobutton(self.instances['window'], text="z-score", variable=self.var_norm , value='z-score') 
        radio1_view.place(relx=pos['rx1'], rely=pos['ry1'], anchor=tk.CENTER)
        radio2_view.place(relx=pos['rx2'], rely=pos['ry2'], anchor=tk.CENTER)   



    def view_metric_radios(self, default_metrics='Euclidiana', metrics = {'Euclidiana': {'rx':0.065,'ry':0.11,'tag':'euclidean'},'Coseno':{'rx':0.065,'ry':0.14,'tag':'cosine'}}):
        self.var_metric = tk.StringVar(self.instances['window'], metrics[default_metrics]['tag'])
        metrics_ = {}
        for m in metrics.keys():
            metrics_[m] = tk.Radiobutton(self.instances['window'], text=m, variable=self.var_metric , value=metrics[m]['tag'])
            metrics_[m].place(relx=metrics[m]['rx'], rely=metrics[m]['ry'], anchor=tk.CENTER)


    def view_linkage_radios(self, linkage = {'Ward': {'rx':0.065,'ry':0.21,'tag':'ward'},'Completo':{'rx':0.065,'ry':0.24,'tag':'complete'},'Promedio':{'rx':0.065,'ry':0.27,'tag':'average'},'Simple':{'rx':0.065,'ry':0.3,'tag':'single'}}):
        self.var_linkage = tk.StringVar(self.instances['window'], linkage['Ward']['tag'])
        linkage_ = {}
        for m in linkage.keys():
            linkage_[m] = tk.Radiobutton(self.instances['window'], text=m, variable=self.var_linkage , value=linkage[m]['tag'])
            linkage_[m].place(relx=linkage[m]['rx'], rely=linkage[m]['ry'], anchor=tk.CENTER)


    def labels_options(self, instances, dic_positions):
        self.var_labels_view = tk.StringVar(instances['view'], "2")
        radio1_view = tk.Radiobutton(instances['view'], text="Todas las etiquetas", variable=self.var_labels_view , value=1, command=instances['fn'])
        radio2_view = tk.Radiobutton(instances['view'], text="Filtrar etiquetas", variable=self.var_labels_view , value=2, command=instances['fn']) 
        radio1_view.grid(row=dic_positions['r0'], column=dic_positions['c0'], sticky=dic_positions['s0'])
        radio2_view.grid(row=dic_positions['r1'], column=dic_positions['c1'], sticky=dic_positions['s1'])      

    def clustering_options(self, instances, dic_positions):
        self.var_clustering_view = tk.StringVar(instances['view'], "2")
        radio1_clus_view = tk.Radiobutton(instances['view'], text="Ver clustering", variable=self.var_clustering_view, value=1, command=instances['fn'])
        radio2_clus_view = tk.Radiobutton(instances['view'], text="Sin clustering", variable=self.var_clustering_view, value=2, command=instances['fn'])  
        radio1_clus_view.grid(row=dic_positions['r0'], column=dic_positions['c0'], sticky=dic_positions['s0'])
        radio2_clus_view.grid(row=dic_positions['r1'], column=dic_positions['c1'], sticky=dic_positions['s1'])  
    

    def view_radios_btn(self, instances, pos={}, type_='clustering'):
        label_clus_label_plot = tk.Label(instances['view'], text="Define etiqueta: ", font="Arial 14")
        self.radio_var_label_plot = tk.StringVar(instances['view'], "1")
        radio1_label_plot = tk.Radiobutton(instances['view'], text="Ninguno", variable=self.radio_var_label_plot, value=1, command=instances['fn'])
        radio2_label_plot = tk.Radiobutton(instances['view'], text="Índice", variable=self.radio_var_label_plot, value=2, command=instances['fn'])
        radio3_label_plot = tk.Radiobutton(instances['view'], text="Etiqueta", variable=self.radio_var_label_plot, value=3, command=instances['fn'])
        radio4_label_plot = tk.Radiobutton(instances['view'], text="Cluster", variable=self.radio_var_label_plot, value=4, command=instances['fn'])

        if type_== 'clustering':
            # Colocar los radio buttons en la self.instances['window']
            label_clus_label_plot.place(relx=pos['lx1'], rely=pos['ly1'], anchor=tk.CENTER)
            radio1_label_plot.place(relx=pos['rx1'], rely=pos['ry1'])
            radio2_label_plot.place(relx=pos['rx2'], rely=pos['ry2'])
            radio3_label_plot.place(relx=pos['rx3'], rely=pos['ry3'])
            radio4_label_plot.place(relx=pos['rx4'], rely=pos['ry4'])
            #label_clus_label_plot.place(relx=0.07, rely=0.25, anchor=tk.CENTER)
            #radio1_label_plot.place(relx=0.018, rely=0.30)
            #radio2_label_plot.place(relx=0.018, rely=0.335)
            #radio3_label_plot.place(relx=0.018, rely=0.37)
            #radio4_label_plot.place(relx=0.018, rely=0.405)
        else:
            label_clus_label_plot.grid(row=pos['r0'], column=pos['c0'], sticky=pos['s0'])
            radio1_label_plot.grid(row=pos['r1'], column=pos['c1'], sticky=pos['s1'])
            radio2_label_plot.grid(row=pos['r2'], column=pos['c2'], sticky=pos['s2'])
            radio3_label_plot.grid(row=pos['r3'], column=pos['c3'], sticky=pos['s3'])
            radio4_label_plot.grid(row=pos['r4'], column=pos['c4'], sticky=pos['s4'])


    def view_btn(self, btn_text, fn, config = {'x': 0.065, 'y':0.5}):
        btn_k_means = tk.Button(self.instances['window'], text=btn_text, command=fn)
        btn_k_means.place(relx=config['x'], rely=config['y'], anchor=tk.CENTER)