import tkinter as tk

class Metrics_Interface:
    def __init__(self, instances):
        self.instances = instances
        self.layer = int(self.instances['entry_layer'].get())
        self.cluster_instance = self.instances['kmeans_instance']
        self.algorithm = self.instances['radio_clus_var'].get()
        analysis_type = self.instances['radio_clus_ana_var'].get()

    def validate_algorithm(self):
        self.layer = int(self.instances['entry_layer'].get())
        self.algorithm = self.instances['radio_clus_var'].get()
        if self.algorithm == "1":
            self.cluster_instance = self.instances['kmeans_instance']
        if self.algorithm == "2":
            self.cluster_instance = self.instances['dbscan_instance']
        if self.algorithm == "3":
            self.cluster_instance = self.instances['meanshift_instance']
        if self.algorithm == "4":
            self.cluster_instance = self.instances['agglomerative_instance']
        if self.algorithm == "5":
            self.cluster_instance = self.instances['spectral_instance']
        if self.algorithm == "6":
            self.cluster_instance = self.instances['gaussian_mixture_instance']

    def extrinsic_Measures(self):
        self.validate_algorithm()
        label= tk.Label(self.instances['window'], text="Métricas extrínsecas: ", font="Arial 16")
        label.place(relx=0.5, rely=0.65, anchor=tk.CENTER)
        score_ARI = self.cluster_instance.data.ARIMetric(True)
        self.block_measure("ARI (Adjusted Rand Index): " + str(round(score_ARI,4)), 0.5, 0.7)
        score_rand = self.cluster_instance.data.rand_Metric(True)
        self.block_measure("Rand index: " + str(round(score_rand,4)), 0.5, 0.75)
        (score_homo, score_com, score_v) = self.cluster_instance.data.homogeneity_completeness_v_measureMetric(True)
        self.block_measure(f"Homogeneidad: {str(round(score_homo,4))},   Completud: {str(round(score_com,4))},   V-Measure: {str(round(score_v,4))}", 0.5, 0.8)
        score_mutual_info = self.cluster_instance.data.mutual_infoMetric(True)
        self.block_measure("Mutual Information: " + str(round(score_mutual_info,4)), 0.5, 0.85)
        score_fowlkes_mallows = self.cluster_instance.data.fowlkes_mallowsMetric(True)
        self.block_measure("Fowlkes Mallows: " + str(round(score_fowlkes_mallows,4)), 0.5, 0.9)


    def intrinsic_Measures(self):
        self.validate_algorithm()
        self.algorithm = self.instances['radio_clus_var'].get()
        label= tk.Label(self.instances['window'], text="Métricas intrínsecas: ", font="Arial 16")
        label.place(relx=0.5, rely=0.25, anchor=tk.CENTER)
        score_silhouette = self.cluster_instance.data.silhouetteScoreMetric(self.layer)
        self.block_measure("Índice de Silueta: " + str(round(score_silhouette,4)), 0.5, 0.3)
        score_davis = self.cluster_instance.data.davisBouldinMetric(self.layer)
        self.block_measure("Índice de Davies-Bouldin: " + str(round(score_davis,4)), 0.5, 0.35)
        score_calinski = self.cluster_instance.data.calinskiMetric(self.layer)
        self.block_measure("Índice de Calinski Harabasz: " + str(round(score_calinski,4)), 0.5, 0.4)
        if self.algorithm == "1":
            score_inertia = self.cluster_instance.data.clusters_obj.inertia_
            self.block_measure("Inercia: " + str(round(score_inertia,4)), 0.5, 0.45)
        if self.algorithm == "6":
            score_bic = self.cluster_instance.data.clusters_obj.data['bic']
            self.block_measure("Criterio de Información Bayesiano (BIC): " + str(round(score_bic,4)), 0.5, 0.45)
            score_aic = self.cluster_instance.data.clusters_obj.data['aic']
            self.block_measure("Criterio de Información de Akaike (AIC): " + str(round(score_aic,4)), 0.5, 0.5)

    def view_radio_measure(self):
        label= tk.Label(self.instances['window'], text="Tipo de métrica: ", font="Arial 14")
        label.place(relx=0.32, rely=0.07, anchor=tk.CENTER)
        self.radio_var_label = tk.StringVar(self.instances['window'], "-1")
        radio1_label = tk.Radiobutton(self.instances['window'], text="Intrínseca", variable=self.radio_var_label, value=1, command=self.intrinsic_Measures)
        radio2_label = tk.Radiobutton(self.instances['window'], text="Extrínseca", variable=self.radio_var_label, value=2, command=self.extrinsic_Measures)
        radio3_label = tk.Radiobutton(self.instances['window'], text="Barrido K", variable=self.radio_var_label, value=3, command=self.instances['go_to_sweeping_k'])
        # Colocar los radio buttons en la self.instances['window']
        radio1_label.place(relx=0.6, rely=0.05)
        radio2_label.place(relx=0.6, rely=0.1)
        radio3_label.place(relx=0.6, rely=0.15)


    def block_measure(self, measure_text, x, y):
        label_measure = tk.Label(self.instances['window'], text=measure_text, font="Arial 14")
        label_measure.place(relx=x, rely=y, anchor=tk.CENTER)
