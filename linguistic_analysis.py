
import tkinter as tk
from plot_interface import Plot_Interface
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict
from tkinter import messagebox
import numpy as np
import re
import nltk
import spacy
nlp_en = spacy.load("en_core_web_sm")
from collections import defaultdict
#import spacy
#nlp_en = spacy.load("en_core_web_sm")
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stopwords = nltk.corpus.stopwords.words('english')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from plotting  import *


class Linguistic_Analysis:
    def __init__(self, instances):
        self.instances = instances
        self.layer = int(self.instances['entry_layer'].get())
        self.algorithm = self.instances['radio_clus_var'].get()
        self.current_instance = self.instances['kmeans_instance']
        self.instance_clusters = self.instances['kmeans']
        self.algorithm_name = 'k-means'
        self.structures_labels = []
        self.analysis_type = '1'


    def update_view(self):
        # Ocultar todos los conjuntos de botones
        self.set1.place_forget()
        self.set2.place_forget()
        self.set3.place_forget()
        self.set4.place_forget()

        # Mostrar el set de botones correspondiente a la selección
        seleccion = self.radio_var.get()
        self.set_instances()
        if seleccion == 1:
            self.set1.place(x=10, y=60)
            self.semantic_analysys_view()
        elif seleccion == 2:
            self.set2.place(x=10, y=60)
            self.gramatical_structure_view()
        elif seleccion == 3:
            self.set3.place(x=10, y=60)
            self.len_sequence_view()
        elif seleccion == 4:
            self.set4.place(x=10, y=60)
            self.semantic_similarity_view()

    def analysis_options(self):
        # Variable para almacenar la selección del radio button
        self.radio_var = tk.IntVar(self.instances['window'], -1)

        # Crear radio buttons
        radio1 = tk.Radiobutton(self.instances['window'], text="Análisis Semántico", variable=self.radio_var, value=1, command=self.update_view)
        radio2 = tk.Radiobutton(self.instances['window'], text="Estructura gramatical", variable=self.radio_var, value=2, command=self.update_view)
        radio3 = tk.Radiobutton(self.instances['window'], text="Tamaño de sentencia", variable=self.radio_var, value=3, command=self.update_view)
        radio4 = tk.Radiobutton(self.instances['window'], text="Similitud semántica", variable=self.radio_var, value=4, command=self.update_view)

        # Ubicar los radio buttons en la ventana
        radio1.place(relx=0.2, rely=0.04, anchor=tk.CENTER)
        radio2.place(relx=0.4, rely=0.04, anchor=tk.CENTER)
        radio3.place(relx=0.6, rely=0.04, anchor=tk.CENTER)
        radio4.place(relx=0.8, rely=0.04, anchor=tk.CENTER)

        # Crear conjuntos de botones
        self.set1 = tk.Frame(self.instances['window'])
        self.set2 = tk.Frame(self.instances['window'])
        self.set3 = tk.Frame(self.instances['window'])
        self.set4 = tk.Frame(self.instances['window'])


    def set_instances(self):
        self.type_analysis = self.instances['radio_embedding_type_analysis'].get()
        self.method = self.instances['radio_var'].get()
        if self.method == '1':
            self.layer = int(self.instances['pca'].layer)
            algorithm = self.instances['pca'].radio_clus_var.get()
        else:
            self.layer = int(self.instances['entry_layer'].get())
            algorithm = self.instances['radio_clus_var'].get()

        if algorithm == "1":
            self.current_instance = self.instances['kmeans_instance']
            self.instance_clusters = self.instances['kmeans']
            self.scaled = self.current_instance.scaled
            self.algorithm_name = 'k-means'
        elif algorithm == "2":
            self.current_instance = self.instances['dbscan_instance']
            self.instance_clusters = self.instances['dbscan']
            self.scaled = self.current_instance.scaled
            self.algorithm_name = 'DBSCAN'
        elif algorithm == "3":
            self.current_instance = self.instances['meanshift_instance']
            self.instance_clusters = self.instances['meanshift']
            self.scaled = self.current_instance.scaled
            self.algorithm_name = 'Mean-shift'
        elif algorithm == "4":
            self.current_instance = self.instances['agglomerative_instance']
            self.instance_clusters = self.instances['agglomerative']
            self.scaled = self.current_instance.scaled
            self.algorithm_name = 'Aglomerativo'
        elif algorithm == "5":
            self.current_instance = self.instances['spectral_instance']
            self.instance_clusters = self.instances['spectral']
            self.scaled = self.current_instance.scaled
            self.algorithm_name = 'Espectral'
        elif algorithm == "6":
            self.current_instance = self.instances['gaussian_mixture_instance']
            self.instance_clusters = self.instances['gaussian_mixture']
            self.scaled = self.current_instance.scaled
            self.algorithm_name = 'Mezcla Gausiana'


    def analysis(self):
        self.analysis_type = str(self.radio_var.get())
        self.set_instances()
        if self.analysis_type == "1": #semantic analysis
            words = self.entry_key_words.get()
            self.words_arr = str(words).split(',')
            self.seqs, self.idxs = self.print_sequences(self.words_arr)
            points = np.array(self.plot_highlight_samples_of_interest(self.idxs))
            self.define_plot_view(self.words_arr, points)
            self.breakdown(self.words_arr)
        elif self.analysis_type == "2": #structure analysis
            self.idxs = self.get_gramatical_structures_analysis()
            #self.define_structures_similarity_labels()
            #structures_legends = [f"Estructura {i}" for i in list(range(1,len(self.idxs) + 1))]
            structures_legends = [f"Estructura {i}" for i in list(range(1,len(self.idxs) + 1))]
            self.print_sequences_structure(structures_legends)
            points = np.array(self.plot_highlight_samples_of_interest(self.idxs))
            self.define_plot_view(structures_legends, points)
            self.breakdown(structures_legends)
        elif self.analysis_type == "3":
            self.idxs, dic_idxs, arr_num, dic_range, arr_legends = self.len_sequences()
            #structures_legends = [f"Tamaño {i}" for i in arr_legends]
            structures_legends = [f"Tamaño {i}" for i in arr_legends]
            self.print_sequences_len(dic_idxs, dic_range, arr_num)
            points = np.array(self.plot_highlight_samples_of_interest(self.idxs))
            self.define_plot_view(structures_legends, points)
            self.breakdown(structures_legends)
        elif self.analysis_type == "4":
            self.idxs, legends = self.get_idxs()
            points = np.array(self.plot_highlight_samples_of_interest(self.idxs))
            self.define_plot_view(legends, points)


    def get_idxs(self): 
        range_sim_ = str(self.entry_range_sim.get()).split(',') 
        #legends = [f'Similitud {string}' for string in range_sim_]
        legends = [f'Similitud {string}' for string in range_sim_]
        labels = self.current_instance.data.labels
        idxs = []
        for range_ in range_sim_:
            idx = []
            if '-' in range_:
                range_sim = range_.split('-')
                for i, label in enumerate(labels.values()):
                    if label >= float(range_sim[0]) and label < float(range_sim[1]):
                        idx.append(i)
            else:
                for i, label in enumerate(labels.values()):
                    if label == float(range_):
                        idx.append(i)
            idxs.append(idx)
        return idxs, legends
            


    def define_plot_view(self, legends, points):
        lab = self.plot_.var_labels_view.get()
        clus = self.plot_.var_clustering_view.get()
        #Filtrar etiquetas y sin clustering
        if clus == "2":
            self.filter_labels_and_without_clustering(lab, legends, points)
        #Filtrar etiquetas y con clustering
        elif clus == "1":
            self.filter_labels_and_with_clustering(lab, legends, points)


    def filter_labels_and_without_clustering(self, label, legends_, highlight_points):
        X = self.current_instance.data.vectors_per_layer[self.layer]
        labels = self.current_instance.data.labels
        #legends = ['Resto'] + legends_
        legends = ['Resto'] + legends_
        label_option = self.get_label_option()
        isRestrictLabel = self.is_restrict_labels(label)
        n = 1 if len(np.unique(self.current_instance.data._labels)) == 1 else 0
        if self.scaled == 'z-score':
            #highlight_linguistic_plot(self.instances, self.current_instance.data.vectors_z_score, highlight_points, labels, label_option, 'Capa ' + str(self.layer+1), legends, len(legends_), n, isRestrictLabel)
            highlight_linguistic_plot(self.instances, self.current_instance.data.vectors_z_score, highlight_points, labels, label_option, 'Capa ' + str(self.layer+1), legends, len(legends_), n, isRestrictLabel)
        else:
            #highlight_linguistic_plot(self.instances, X, highlight_points, labels, label_option, 'Capa ' + str(self.layer+1), legends, len(legends_), n, isRestrictLabel)
            highlight_linguistic_plot(self.instances, X, highlight_points, labels, label_option, 'Capa ' + str(self.layer+1), legends, len(legends_), n, isRestrictLabel)


    def filter_labels_and_with_clustering(self, label, legends_, points):
        X = self.current_instance.data.vectors_per_layer[self.layer]
        labels = self.current_instance.data.labels
        n = 1
        if min(self.current_instance.data._labels) == -1: 
            #legends = list(range(1,len(np.unique(self.current_instance.data._labels)))) + ['Valores atípicos'] + legends_
            legends = list(range(1,len(np.unique(self.current_instance.data._labels)))) + ['Atypical values'] + legends_
        else:
            legends = list(range(1,len(np.unique(self.current_instance.data._labels))+1)) + legends_

        label_option = self.get_label_option()
        isRestrictLabel = self.is_restrict_labels(label)
        if self.scaled == 'z-score':
            #highlight_linguistic_plot(self.instances, self.current_instance.data.vectors_z_score, points, labels, label_option, 'Capa ' + str(self.layer +1), legends, len(legends_), n, isRestrictLabel)
            highlight_linguistic_plot(self.instances, self.current_instance.data.vectors_z_score, points, labels, label_option, 'Capa ' + str(self.layer +1), legends, len(legends_), n, isRestrictLabel)
        else:
            #highlight_linguistic_plot(self.instances, X, points, labels, label_option, 'Capa ' + str(self.layer +1), legends, len(legends_), n, isRestrictLabel)
            highlight_linguistic_plot(self.instances, X, points, labels, label_option, 'Capa ' + str(self.layer +1), legends, len(legends_), n, isRestrictLabel)


    def is_restrict_labels(self, label):
        if label == "2":
            return True
        else: 
            return False

    def get_label_option(self):
        label_option = self.plot_.radio_var_label_plot.get()
        if label_option == "1":
            return ''
        elif label_option == "2":
            return 'index'
        elif label_option == "3":
            return 'label'
        elif label_option == "4":
            return 'cluster'


    def plot_highlight_samples_of_interest(self, data):
        interest_points = {}
        points, data_labels = [], []
        #n_clusters = len(np.unique(self.current_instance.data._labels))
        n_max = max(self.current_instance.data._labels)
        if min(self.current_instance.data._labels) == -1:
            for i, d in enumerate(self.current_instance.data._labels):
                if d == -1:
                    data_labels.append(n_max+1)
                else:
                    data_labels.append(d)
        else:
            data_labels = self.current_instance.data._labels

        n_max_ = max(data_labels)

        for i, id in enumerate(data):
            for j in id: #j es el índice
                #interest_points[j] = n_clusters+i+1
                interest_points[j] = n_max_+i+1
        dic = dict(sorted(self.current_instance.data.sequences.items(), key=lambda item: item[0]))
        
        for (n,_) in dic.items():
            if n in interest_points:
                points.append(int(interest_points[n]))
            else:
                if self.plot_.var_clustering_view.get() == "2":
                    points.append(0)
                elif self.plot_.var_clustering_view.get() == "1":
                    #points.append(self.current_instance.data._labels[n])
                    points.append(data_labels[n])
        return points

    def plot_fn(self, frame):
        #Plot
        self.instances['frame'] = frame
        self.instances['frame'].grid_rowconfigure(1, weight=1)
        self.instances['frame'].grid_columnconfigure(2, weight=1)
        self.instances['frame'].grid_columnconfigure(3, weight=1)
        self.instances['frame'].grid_columnconfigure(4, weight=1)
        self.instances['frame'].grid_columnconfigure(6, weight=1)

        for i in range(2,24):
            self.instances['frame'].grid_rowconfigure(i, weight=1)
        self.plot_ = Plot_Interface(self.instances,[1.2],(6,6),True)
        self.plot_.view_plot()

        self.instances['plot'] = self.plot_.plot
        self.instances['canvas'] = self.plot_.canvas
        self.instances['canvas_widget'] =self.plot_.canvas_widget
        self.instances['zoom_factor'] =self.plot_.zoom_factor
        self.instances['zoom_direction_button'] = self.plot_.btn_zoom_direction
        self.instances['canvas'].draw()
        self.instances['canvas_widget'].grid(row=0, column=6, rowspan=20, padx=10, pady=5, columnspan=5)
        self.instances['canvas_widget'].config(width=800, height=540)

    def semantic_analysys_view(self):
        #searcher and buttons
        label_key_words = tk.Label(self.set1, text="Palabras Clave: ", font="Arial 14")
        self.entry_key_words = tk.Entry(self.set1, width=35)
        btn_find = tk.Button(self.set1, text="Buscar", command=self.analysis)
        label_key_words.grid(row=0, column=0, padx=10, pady=10, sticky="n")
        self.entry_key_words.grid(row=0, column=1, padx=10, pady=10, columnspan=4, sticky="n")
        btn_find.grid(row=1, column=0, padx=5, pady=5, sticky="n")
        #Plot
        self.plot_fn(self.set1)

        self.show_sentences(self.set1)
        type_labels={'r0':9, 'c0':3, 's0':'ne', 'r1':10,'c1':3,'s1':'ne','r2':11,'c2':3,'s2':'ne','r3':12,'c3':3,'s3':'ne','r4':13,'c4':3,'s4':'ne',}
        self.plot_.view_radios_btn({'view': self.set1, 'fn':self.analysis},type_labels, '')
        self.plot_.labels_options({'view': self.set1, 'fn':self.analysis},{'r0':2, 'c0':3, 's0':'ne', 'r1':3,'c1':3, 's1':'ne',})
        self.plot_.clustering_options({'view': self.set1, 'fn':self.analysis},{'r0':5, 'c0':3, 's0':'ne', 'r1':6,'c1':3, 's1':'ne'})
        self.show_topics_per_cluster()
        self.show_breakdown(self.set1)


    def print_sequences(self, words_arr):
        self.canvas_training.delete("all")
        seqs, idxs, rest = [], [], []
        vertical_position = 20
        for w in words_arr:
            sequences_with_key_word, idx = [], []
            self.canvas_training.create_text(25, vertical_position, text=f"{w}: ", anchor="w")
            vertical_position += 20
            for (n,s) in self.current_instance.data.sequences.items():
                if w in s:
                    sequences_with_key_word.append((n,s))
                    idx.append(n)
                    self.canvas_training.create_text(25, vertical_position, text=f"{n} : similaridad: {self.current_instance.data.labels[n]}, dim: {self.current_instance.data.dimensions[n]}, secuencia: {s}", anchor="w")
                    #self.canvas_training.create_text(10, vertical_position, text=f"{n}, similarity: {self.current_instance.data.labels[n]}, sequence:  {s}")
                    vertical_position += 20
            seqs.append(sequences_with_key_word)
            idxs.append(idx)
        self.canvas_training.update_idletasks()
        self.canvas_training.configure(scrollregion=self.canvas_training.bbox("all"))
        return seqs, idxs

    
    def breakdown(self, legends):
        self.canvas_breakdown.delete("all")
        cnt, total = {}, {}
        self.clusters_idx = self.define_clusters()
        for j, idxs in enumerate(self.idxs):
            for k, i in enumerate(idxs):
                for c, arr in self.clusters_idx.items():
                    if i in arr:
                        if (legends[j], c) in cnt:
                            cnt[(legends[j], c)].append(i)
                        else:
                            cnt[(legends[j], c)] = [i]
                    else:
                        if k > 0:
                            continue
                        else:
                            cnt[(legends[j], c)] = []

        for (_,clus), v in cnt.items():
            if clus in total:
                total[clus] = total[clus] + v
            else: 
                total[clus] = v
        pos = 5
        for c_, arr_ in self.clusters_idx.items():
            self.canvas_breakdown.create_text(25, pos, text=f"El cluster {c_+1} tiene {len(arr_)} muestras", anchor="w")
            pos += 25
            for w in legends:
                if self.analysis_type == "1":
                    self.canvas_breakdown.create_text(25, pos, text=f"El {round((len(cnt[(w, c_)])/len(arr_)*100),3)}% del cluster {c_} tiene la palabra {w} ({len(cnt[(w, c_)])})", anchor="w")
                elif self.analysis_type == "2":
                    self.canvas_breakdown.create_text(25, pos, text=f"El {round((len(cnt[(w, c_)])/len(arr_)*100),3)}% del cluster {c_} tiene la {w} ({len(cnt[(w, c_)])})", anchor="w")
                elif self.analysis_type == "3":
                    self.canvas_breakdown.create_text(25, pos, text=f"El {round((len(cnt[(w, c_)])/len(arr_)*100),3)}% del cluster {c_} tiene secuencias de {w} ({len(cnt[(w, c_)])})", anchor="w")
                pos += 20

            self.canvas_breakdown.create_text(25, pos, text=f'El {round((len(total[(c_)])/len(arr_)*100),4)}% del cluster {c_} tiene "{" ".join(legends)}"', anchor="w")
            pos += 30
            self.canvas_breakdown.update_idletasks()
            self.canvas_breakdown.configure(scrollregion=self.canvas_breakdown.bbox("all"))



    def define_clusters(self):
        clusters_idx = {}
        for i, label in enumerate(self.current_instance.data._labels):
            if label in clusters_idx:
                clusters_idx[label].append(i)
            else:
                clusters_idx[label] = [i]
        return clusters_idx


    def print_topics(self):
        clusters = {}
        num_topics = int(self.var_topics_num.get())
        num_words_per_topic = int(self.var_words_in_topics_num.get())
        max_df = float(self.var_max_df.get())
        seqs = self.process_sentences(self.current_instance.data.sequences)
        for i, label in enumerate(self.current_instance.data._labels):
            if label in clusters:
                clusters[label].append(seqs[i])
            else:
                clusters[label] = [seqs[i]]
        vertical_position = 20
        self.canvas_bottom.delete("all")
        for k, v in clusters.items():
            _, pos = self.find_topics(v, str(k+1), num_topics, vertical_position, num_words_per_topic, max_df)
            vertical_position = pos + 10


    def process_sentences(self,sentences,has_stopwords=False,lemmatize=True,has_numbers=False,has_punctuation=False):
        sents = []
        lemmatizer = WordNetLemmatizer()
        processed_text = []
        for i, text in sentences.items():
            text = text.replace('s1:', '')
            text = text.replace('s2:', '')
            if not has_numbers:
                text = re.sub(r'\d+', '', text)
            if not has_punctuation:
                text = text.translate(str.maketrans('', '', string.punctuation))
            tokens = word_tokenize(text.lower())
            if not has_stopwords:
                tokens = [token for token in tokens if token not in stopwords]
            if lemmatize:
                #text = ' '.join(tokens)
                #doc = nlp_en(text)
                #tokens = [token.lemma_ for token in doc]
                tokens = [lemmatizer.lemmatize(token) for token in tokens]
            processed_text.append(' '.join(list(OrderedDict.fromkeys(tokens)))) #eliminar palabras repetidas en misma secuencia
        return processed_text


    def find_topics(self, preprocessed_sents, cluster= '0', num_topics = 5, vertical_position = 20, words_in_topics = 3, max_df=0.15, min_df=1):
        stopwords_ = stopwords + str(self.entry_stopwords.get()).split(',')
        # Inicializar el vectorizador TF-IDF
        #vectorizer = CountVectorizer(stop_words=stopwords_, max_df=max_df, min_df=min_df)
        vectorizer = TfidfVectorizer(stop_words=stopwords_, max_df=max_df, min_df=min_df)
        # Calcular las características TF-IDF para las oraciones
        term_counts = vectorizer.fit_transform(preprocessed_sents)
        #print(features)
        # Definir el modelo LDA y ajustarlo al conjunto de sentencias
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(term_counts)
        # Obtener los tópicos principales
        #lda.components_ es una matriz que contiene la distribución de palabras en cada tópico. Cada fila = tópico y cada columna = una palabra en vocabulario.
        #argsort(axis=1) ordena los índices de las palabras en cada fila (tópico) en orden ascendente.
        #[:, ::-1] invierte el orden de los índices en cada fila (tópico), de modo que las palabras más relevantes van primero.
        top_topics = lda.components_.argsort(axis=1)[:, ::-1]
        feature_names = vectorizer.get_feature_names_out()
        #print(top_topics) #top_topics it is an array (num_topics, words)
        topics_ = []
        pos = vertical_position
        self.canvas_bottom.create_text(25, pos, text=f"Cluster {cluster}: ", anchor="w")
        print(f"Cluster {cluster}: ")
        pos += 20
        for topic_idx, topic in enumerate(top_topics):
            top_words = [feature_names[i] for i in topic[:words_in_topics]]  # Las palabras más relevantes del tópico
            topics_.append(top_words)
            #print(f"Tópico {topic_idx + 1}: {' '.join(top_words)}")
            self.canvas_bottom.create_text(25, pos, text=f"Topico {topic_idx + 1} : {' '.join(top_words)}", anchor="w")
            print(f"Topico {topic_idx + 1} : {' '.join(top_words)}")
            pos += 20
        self.canvas_bottom.update_idletasks()
        self.canvas_bottom.configure(scrollregion=self.canvas_bottom.bbox("all"))
        return topics_, pos
    

    def show_topics_per_cluster(self):
        label_topics = tk.Label(self.set1, text="Tópicos por cluster: ", font="Arial 14")
        label_topics.grid(row=15, column=0, padx=10, sticky="n")
        self.var_topics_num = tk.StringVar(self.set1, "5")
        self.entry_topics = tk.Entry(self.set1, width=8, textvariable=self.var_topics_num)
        self.entry_topics.grid(row=15, column=1, padx=10, sticky="n")

        label_wordsInTopics = tk.Label(self.set1, text="Palabras por tópico: ", font="Arial 14")
        label_wordsInTopics.grid(row=16, column=0, padx=10, sticky="n")
        self.var_words_in_topics_num = tk.StringVar(self.set1, "3")
        self.entry_words_in_topics = tk.Entry(self.set1, width=8, textvariable=self.var_words_in_topics_num)
        self.entry_words_in_topics.grid(row=16, column=1, padx=10, sticky="n")

        label_max_df = tk.Label(self.set1, text="Max Doc. Freq. (%): ", font="Arial 14")
        label_max_df.grid(row=17, column=0, padx=10, sticky="n")
        self.var_max_df = tk.StringVar(self.set1, "0.8")
        self.entry_max_df = tk.Entry(self.set1, width=8, textvariable=self.var_max_df)
        self.entry_max_df.grid(row=17, column=1, padx=10, sticky="n")

        label_stopwords = tk.Label(self.set1, text=" + Stopwords: ", font="Arial 14")
        label_stopwords.grid(row=18, column=0, padx=10, sticky="n")
        self.entry_stopwords = tk.Entry(self.set1, width=25)
        self.entry_stopwords.grid(row=18, column=1, padx=10, sticky="n")

        btn_find_topics = tk.Button(self.set1, text="Buscar", command=self.print_topics)
        btn_find_topics.grid(row=19, column=0, padx=10, sticky="nw")

        self.show_bottom_canvas(self.instances['frame'])


    def show_sentences(self, frame):
        # Crear un frame para contener el Canvas
        frame_training = tk.Frame(frame)
        frame_training.grid(row=22, column=6, pady=5, columnspan=5, sticky="nsew")

        # Crear un Canvas dentro del frame
        self.canvas_training = tk.Canvas(frame_training, width=790, height=110)
        self.canvas_training.grid(row=0, column=0, sticky="nsew")

        # Agregar una barra de desplazamiento vertical
        scrollbar_y = tk.Scrollbar(frame_training, command=self.canvas_training.yview)
        scrollbar_y.grid(row=0, column=1, columnspan=5, sticky="ns")
        self.canvas_training.configure(yscrollcommand=scrollbar_y.set)

        # Agregar una barra de desplazamiento horizontal
        scrollbar_x = tk.Scrollbar(frame_training, orient="horizontal", command=self.canvas_training.xview)
        scrollbar_x.grid(row=23, column=0, columnspan=5, sticky="ew")
        self.canvas_training.configure(xscrollcommand=scrollbar_x.set)

        self.canvas_training.bind("<Configure>", lambda event: self.configurar_vista(event))


    def show_breakdown(self, frame):
        # Crear un frame para contener el Canvas
        frame_breakdown = tk.Frame(frame)
        frame_breakdown.grid(row=2, column=0, pady=5, columnspan=2, rowspan=9, sticky="nsew")

        # Crear un Canvas dentro del frame
        self.canvas_breakdown = tk.Canvas(frame_breakdown, width=350, height=190)
        self.canvas_breakdown.grid(row=0, column=0, sticky="nsew")

        # Agregar una barra de desplazamiento vertical
        scrollbar_y = tk.Scrollbar(frame_breakdown, command=self.canvas_breakdown.yview)
        scrollbar_y.grid(row=0, column=2, columnspan=2, rowspan=9, sticky="ns")
        self.canvas_breakdown.configure(yscrollcommand=scrollbar_y.set)

        # Agregar una barra de desplazamiento horizontal
        scrollbar_x = tk.Scrollbar(frame_breakdown, orient="horizontal", command=self.canvas_breakdown.xview)
        scrollbar_x.grid(row=11, column=0, columnspan=2, rowspan=9, sticky="ew")
        self.canvas_breakdown.configure(xscrollcommand=scrollbar_x.set)

        # Configurar la vista desplazable
        self.canvas_breakdown.bind("<Configure>", lambda event: self.configurar_vista(event))


    def configurar_vista(self, event):
        self.canvas_training.configure(scrollregion=self.canvas_training.bbox("all"))


    def gramatical_structure_view(self):

        btn_search_structure = tk.Label(self.set2, text="# Estructuras comunes: ", font="Arial 14")
        btn_search_structure.grid(row=0, column=0, padx=10, pady=10)

        self.var_num_structures = tk.StringVar(self.set2, "5")
        self.entry_num_struc = tk.Entry(self.set2, width=8, textvariable=self.var_num_structures)
        self.entry_num_struc.grid(row=0, column=1, padx=10)

        btn_search_structure = tk.Button(self.set2, text="Buscar estructuras", command=self.analysis)
        btn_search_structure.grid(row=1, column=0, padx=10, pady=10)

        label_structure = tk.Label(self.set2, text="Num. Estructura: ", font="Arial 14")
        label_structure.grid(row=11, column=0, padx=10, pady=10)

        self.var_struc_selected = tk.StringVar(self.set2, "1")
        self.entry_struc_selected = tk.Entry(self.set2, width=8, textvariable=self.var_struc_selected)
        self.entry_struc_selected.grid(row=11, column=1, padx=10)

        label_similarity = tk.Label(self.set2, text="Similaridad (%): ", font="Arial 14")
        label_similarity.grid(row=12, column=0, padx=10, pady=10)

        self.var_sim = tk.StringVar(self.set2, "0.95")
        self.entry_sim = tk.Entry(self.set2, width=8, textvariable=self.var_sim)
        self.entry_sim.grid(row=12, column=1, padx=10)

        btn_show_sim = tk.Button(self.set2, text="Mostrar", command=self.define_structures_similarity_labels)
        btn_show_sim.grid(row=13, column=0, padx=10, pady=10)


        #Plot
        self.plot_fn(self.set2)

        type_labels={'r0':9, 'c0':3, 's0':'ne', 'r1':10,'c1':3,'s1':'ne','r2':11,'c2':3,'s2':'ne','r3':12,'c3':3,'s3':'ne','r4':13,'c4':3,'s4':'ne',}
        self.plot_.view_radios_btn({'view': self.set2, 'fn':self.analysis},type_labels, '')
        self.plot_.labels_options({'view': self.set2, 'fn':self.analysis},{'r0':2, 'c0':3, 's0':'ne', 'r1':3,'c1':3, 's1':'ne',})
        self.plot_.clustering_options({'view': self.set2, 'fn':self.analysis},{'r0':5, 'c0':3, 's0':'ne', 'r1':6,'c1':3, 's1':'ne'})
        self.show_sentences(self.set2)
        self.show_breakdown(self.set2)
        self.show_bottom_canvas(self.set2)

        btn_boxplot = tk.Button(self.set2, text="Boxplot", command=self.instances['go_to_boxs_plots'])
        btn_boxplot.grid(row=0, column=8, sticky="ne")


    def get_gramatical_structure(self, sentence):
        doc = nlp_en(sentence)
        structure = tuple(token.dep_ for token in doc)
        return structure

    def compare_structures(self, sentences):
        structures = defaultdict(list)
        structure_list = []
        for sentence in sentences:
            structure = self.get_gramatical_structure(sentence[0])
            structure_list.append(structure)
            structures[structure].append((sentence[0],sentence[1]))
        self.structure_list = structure_list
        return structures

    def get_gramatical_structures_analysis(self):
        sents, sents_struc = [], []
        most_common = int(self.var_num_structures.get())
        sentences = self.current_instance.data.sequences
        for i, text in sentences.items():
            text = text.replace('s1:', '')
            text = text.replace('s2:', '')
            sents.append((text,i))
        if not hasattr(self, 'grama_structures'):
            self.grama_structures = self.compare_structures(sents)
        sorted_values = sorted([(k,len(v)) for k,v in self.grama_structures.items()], key=lambda x: x[1], reverse=True)[:most_common]
        self.sorted_main_structures = sorted_values
        vertical_position = 20
        self.canvas_bottom.create_text(25, vertical_position, text=f"Estructuras: ", anchor="w")
        vertical_position += 20
        #print("Estructuras: ")
        for i, (text, num) in enumerate(sorted_values):
            self.canvas_bottom.create_text(25, vertical_position, text=f"Hay {num} muestras con la estructura {i+1}: ", anchor="w")
            vertical_position += 20
            self.canvas_bottom.create_text(25, vertical_position, text=f"{text}", anchor="w")
            vertical_position += 20
            #print(f"Hay {num} muestras con la estructura {i+1}: ")
            #print(text)
        # form index list

        self.canvas_bottom.update_idletasks()
        self.canvas_bottom.configure(scrollregion=self.canvas_bottom.bbox("all"))

        gram_s = []
        for (sg,_) in sorted_values:
            idx_list = []
            for (_, i) in self.grama_structures[sg]:
                idx_list.append(i)
            gram_s.append(idx_list)
        return gram_s

    def define_structures_similarity_labels(self):
        n_sent = int(self.entry_struc_selected.get())-1
        sim_limit = float(self.entry_sim.get())
        reference_structure = self.sorted_main_structures[n_sent][0]
        vocabulary = self.get_vocabulary(self.structure_list)
        vectors = self.feature_engineering_freqs(self.structure_list, vocabulary)
        reference = self.feature_engineering_freqs([reference_structure], vocabulary)
        cosine_similarities = [cosine_similarity(reference, [vector])[0, 0] for vector in vectors]
        self.structures_labels = np.array(cosine_similarities).round(3)
        idxs_similarity = []
        for i, similarity in enumerate(self.structures_labels):
            if similarity >= sim_limit:
                idxs_similarity.append(i)
        points = np.array(self.plot_highlight_samples_of_interest([idxs_similarity]))
        self.define_plot_view([f'{sim_limit} similar'], points)
        



    def get_vocabulary(self, texts):
        voc = [word for text in texts for word in text]
        return sorted(set(voc))

    def feature_engineering_freqs(self, texts, vocabulary):
        vocabulary_freqs = []
        for text in texts:
            vector = [] #
            for voc in vocabulary:
            # In vector saves a list of vocabulary's length. 
            # Iterate each vocabulary word and count in each text list
                vector.append(text.count(voc))
            vocabulary_freqs.append(vector)
        return vocabulary_freqs

    """
    def define_structures_similarity_labels(self):
        reference_structure = self.sorted_main_structures[0]
        tfidf_vectorizer = TfidfVectorizer()
        sents = [' '.join(i) for i in self.structure_list]
        tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(reference_structure[0])] + sents)
        cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:len(self.structure_list)+1])
        self.structures_labels = cosine_similarities.round(3)[0]
    """

    def print_sequences_structure(self, struc):
        self.canvas_training.delete("all")
        vertical_position = 20
        for l, arr in enumerate(self.idxs):
            self.canvas_training.create_text(25, vertical_position, text=f"{struc[l]} ({len(arr)} elementos): ", anchor="w")
            vertical_position += 20
            for i in arr:
                self.canvas_training.create_text(25, vertical_position, text=f"{i} : similaridad: {self.current_instance.data.labels[i]}, dim:{self.current_instance.data.dimensions[i]}, secuencia:  {self.current_instance.data.sequences[i]}", anchor="w")
                vertical_position += 20

        self.canvas_training.update_idletasks()
        self.canvas_training.configure(scrollregion=self.canvas_training.bbox("all"))

    def show_bottom_canvas(self, frame):
        # Crear un frame para contener el Canvas
        frame_bottom = tk.Frame(frame)
        frame_bottom.grid(row=22, column=0, pady=5, columnspan=3, sticky="nsew")

        # Crear un Canvas dentro del frame
        self.canvas_bottom = tk.Canvas(frame_bottom, width=390, height=100)
        self.canvas_bottom.grid(row=0, column=0, sticky="nsew")

        # Agregar una barra de desplazamiento vertical
        scrollbar_y = tk.Scrollbar(frame_bottom, command=self.canvas_bottom.yview)
        scrollbar_y.grid(row=0, column=1, columnspan=3, sticky="ns")
        self.canvas_bottom.configure(yscrollcommand=scrollbar_y.set)

        # Agregar una barra de desplazamiento horizontal
        scrollbar_x = tk.Scrollbar(frame_bottom, orient="horizontal", command=self.canvas_bottom.xview)
        scrollbar_x.grid(row=23, column=0, columnspan=3, sticky="ew")
        self.canvas_bottom.configure(xscrollcommand=scrollbar_x.set)

        # Configurar la vista desplazable
        self.canvas_bottom.bind("<Configure>", lambda event: self.configurar_vista(event))


    def len_sequence_view(self):
        self.label_info_len = tk.Label(self.set3, text=" ", font="Arial 15")
        self.label_info_len.grid(row=0, column=0, padx=10, columnspan=4, pady=10, sticky='n')

        label_seqs_len = tk.Label(self.set3, text="Tamaños de secuencia: ", font="Arial 14")
        label_seqs_len.grid(row=1, column=0, padx=10, pady=10)

        self.var_seqs_len = tk.StringVar(self.set3)
        self.entry_seqs_len = tk.Entry(self.set3, width=20, textvariable=self.var_seqs_len)
        self.entry_seqs_len.grid(row=1, column=1, padx=10)

        btn_search_seqs_len = tk.Button(self.set3, text="Buscar", command=self.analysis)
        btn_search_seqs_len.grid(row=1, column=3, padx=10, pady=10)

        #Plot
        self.plot_fn(self.set3)
        self.find_max_min_len()
        type_labels={'r0':9, 'c0':3, 's0':'ne', 'r1':10,'c1':3,'s1':'ne','r2':11,'c2':3,'s2':'ne','r3':12,'c3':3,'s3':'ne','r4':13,'c4':3,'s4':'ne',}
        self.plot_.view_radios_btn({'view': self.set3, 'fn':self.analysis},type_labels, '')
        self.plot_.labels_options({'view': self.set3, 'fn':self.analysis},{'r0':2, 'c0':3, 's0':'ne', 'r1':3,'c1':3, 's1':'ne',})
        self.plot_.clustering_options({'view': self.set3, 'fn':self.analysis},{'r0':5, 'c0':3, 's0':'ne', 'r1':6,'c1':3, 's1':'ne'})
        self.show_sentences(self.set3)
        self.show_breakdown(self.set3)
        
        btn_boxplot = tk.Button(self.set3, text="Boxplot", command=self.instances['go_to_boxs_plots'])
        btn_boxplot.grid(row=0, column=8, sticky="ne")


    def find_max_min_len(self):
        dims_seqs = self.current_instance.data.dimensions
        self.max_dim = max(dims_seqs.values())
        self.min_dim = min(dims_seqs.values())
        sum = 0
        for valor in dims_seqs.values():
            sum += valor
        avr = sum / len(dims_seqs) 
        self.label_info_len.config(text=f"Secuencia mínima: {self.min_dim} items, grande: {self.max_dim} items, media: {round(avr,3)} items")


    def len_sequences(self):
        dims_seq = self.current_instance.data.dimensions
        seqs_len = str(self.var_seqs_len.get()).split(',')
        #seqs_len_legend = [i.replace('-', ' a ') for i in seqs_len]
        seqs_len_legend = [i.replace('-', '-') for i in seqs_len]
        #seqs_len = [int(k) for k in seqs_len]
        arr_seqs_len, dic_seqs_len, dic_range  = [], {}, {}
        for idx, i in enumerate(seqs_len):
            seqs_len_ = []
            if '-' in i:
                l = i.split("-")
                dic_range[idx] = list(range(int(l[0]),int(l[1])+1))
                for n in list(range(int(l[0]),int(l[1])+1)):
                    seqs_len_2 = []
                    if n > self.max_dim or n < self.min_dim:
                        messagebox.showerror("Error", "Introduce valores enteros válidos dentro del rango separados por coma.")
                        break
                    else:
                        for k,v in dims_seq.items():
                            if n == v:
                                seqs_len_.append(k)
                                seqs_len_2.append(k)
                        dic_seqs_len[n] = seqs_len_2
                arr_seqs_len.append(seqs_len_)
            else:
                i = int(i)
                if i > self.max_dim or i < self.min_dim:
                    messagebox.showerror("Error", "Introduce valores enteros válidos dentro del rango separados por coma.")
                    break
                else:
                    for k,v in dims_seq.items():
                        if i == v:
                            seqs_len_.append(k)
                    arr_seqs_len.append(seqs_len_)
                    dic_seqs_len[i] = seqs_len_
        return arr_seqs_len, dic_seqs_len, seqs_len, dic_range, seqs_len_legend


    def print_sequences_len(self, dic_idxs, dic_range, arr_num):
        self.canvas_training.delete("all")
        vertical_position = 20
        if len(dic_range) > 0: #por rango
            for m, (_,v) in enumerate(dic_range.items()):
                self.canvas_training.create_text(25, vertical_position, text=f"Rango de {v[0]} a {v[-1]}, con total de {len(self.idxs[m])} muestras: ", anchor="w")
                vertical_position += 30
                for i in v:
                    self.canvas_training.create_text(25, vertical_position, text=f"Tamaño de secuencia de {i} elementos ({len(dic_idxs[i])}): ", anchor="w")
                    vertical_position += 25
                    for item in dic_idxs[i]:
                        self.canvas_training.create_text(25, vertical_position, text=f"{item} : similaridad: {self.current_instance.data.labels[item]}, dim: {self.current_instance.data.dimensions[item]}, secuencia:  {self.current_instance.data.sequences[item]}", anchor="w")
                        vertical_position += 20
                    vertical_position += 5
                vertical_position += 5
        else: #individual
            for l, arr in enumerate(self.idxs):
                self.canvas_training.create_text(25, vertical_position, text=f"Tamaño de secuencia de  {arr_num[l]}  elementos ({len(arr)}): ", anchor="w")
                vertical_position += 25
                for i in arr:
                    self.canvas_training.create_text(25, vertical_position, text=f"{i} : similaridad: {self.current_instance.data.labels[i]}, dim: {self.current_instance.data.dimensions[i]}, secuencia:  {self.current_instance.data.sequences[i]}", anchor="w")
                    vertical_position += 20
                vertical_position += 5

        self.canvas_training.update_idletasks()
        self.canvas_training.configure(scrollregion=self.canvas_training.bbox("all"))


    def semantic_similarity_view(self):
        label_search_sem_similarity = tk.Label(self.set4, text="Rango de similitud: ", font="Arial 14")
        label_search_sem_similarity.grid(row=0, column=0, padx=10, pady=10)

        self.var_range_sim = tk.StringVar(self.set4, "0-0.5")
        self.entry_range_sim = tk.Entry(self.set4, width=8, textvariable=self.var_range_sim)
        self.entry_range_sim.grid(row=0, column=1, padx=10)

        btn_search_sim = tk.Button(self.set4, text="Buscar", command=self.analysis)
        btn_search_sim.grid(row=1, column=0, padx=10, pady=10)

        #Plot
        self.plot_fn(self.set4)

        type_labels={'r0':9, 'c0':3, 's0':'ne', 'r1':10,'c1':3,'s1':'ne','r2':11,'c2':3,'s2':'ne','r3':12,'c3':3,'s3':'ne','r4':13,'c4':3,'s4':'ne',}
        self.plot_.view_radios_btn({'view': self.set4, 'fn':self.analysis},type_labels, '')
        self.plot_.labels_options({'view': self.set4, 'fn':self.analysis},{'r0':2, 'c0':3, 's0':'ne', 'r1':3,'c1':3, 's1':'ne',})
        self.plot_.clustering_options({'view': self.set4, 'fn':self.analysis},{'r0':5, 'c0':3, 's0':'ne', 'r1':6,'c1':3, 's1':'ne'})

        btn_boxplot = tk.Button(self.set4, text="Boxplot", command=self.instances['go_to_boxs_plots'])
        btn_boxplot.grid(row=0, column=8, sticky="ne")