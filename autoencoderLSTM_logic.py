import torch
import torch.nn as nn
import random
import time
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler


def load_pth(i, tk, messagebox, MODEL_CONFIGURATION):
    root = i['entry_root'].get()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        vectors = torch.load(root, map_location=torch.device(device))
        i['label_valid'].config(text=f"Archivo cargado.")
        try:
            global data_loader
            data_loader = form_tensors(MODEL_CONFIGURATION, vectors, i['embedding_type'].get())
            data_loader_size = len(data_loader)
            i['label_elements'].config(text=f"Se detectaron {str(data_loader_size)} elementos")
            show_train_components(i, tk)
        except Exception as e:
            messagebox.showerror("Error", "Se produjo una excepción:" + str(e))
    except FileNotFoundError:
        # Manejo de la excepción si el archivo no se encuentra
        messagebox.showerror("Error", "El archivo " + root + " no existe.")
    except Exception as e:
        # Manejo de otras excepciones
        print(f"Error al cargar el archivo .pth: {e}")
        messagebox.showerror("Error", "Error al cargar el archivo " + root)

def show_train_components(i, tk):
    i['label_epochs'].place(relx=0.15, rely=0.35, anchor=tk.CENTER)
    i['entry_epochs'].place(relx=0.4, rely=0.35, anchor=tk.CENTER)
    i['label_lr'].place(relx=0.15, rely=0.4, anchor=tk.CENTER)
    i['entry_lr'].place(relx=0.4, rely=0.4, anchor=tk.CENTER)
    i['label_dim_reduction'].place(relx=0.05, rely=0.43)
    i['entry_dim_reduction'].place(relx=0.24, rely=0.425)
    i['label_optim'].place(relx=0.15, rely=0.5, anchor=tk.CENTER)
    i['radiom'].place(relx=0.33, rely=0.475)
    i['btn_train'].place(relx=0.7, rely=0.4, anchor=tk.CENTER)

def show_training_process(i):
    i['frame_training'].place(relx=0.09, rely=0.55)
    i['canvas_training'].pack(side='left', fill='both', expand=True)
    i['scrollbar_training'].pack(side='right', fill='y')

def show_save_embeddings(i):
    i['entry_save_reduction'].place(relx=0.225, rely=0.745, anchor=i['tk'].CENTER)
    i['btn_save_emb_red'].place(relx=0.6, rely=0.74, anchor=i['tk'].CENTER)

def form_tensors(config, tensors, embedding_type):
    layers = config['layers']
    heads = config['heads']
    num_sentences = list(tensors.keys())[-1][0]+1
    attentions_concat_heads = {}
    attentions_matrix_list = []
    vectors_list = []
    if embedding_type == 'Attention': #Autoatenciones
        for i in range(num_sentences):
            for j in range(layers):
                tensor_list = []
                for k in range(heads):
                    tensor_list.append(torch.tensor(tensors[(i,j,k)]['vectors']).flatten())
                #attentions_concat_heads[(i,j)] = torch.stack(tensor_list).unsqueeze(0).permute(0, 2, 1)
                stack = torch.stack(tensor_list)
                attentions_concat_heads[(i,j)] = stack
                attentions_matrix_list.append(stack)
                vectors_list.append(((i,j),tensors[(i,j,k)]['sequence'],tensors[(i,j,k)]['label'],tensors[(i,j,k)]['dimension'],stack))
        vectors_list = [(id,s,label,dim,tensor.permute(1,0)) for id, s, label, dim, tensor in vectors_list]
    if embedding_type == 'CLS': #Token CLS
        for i in range(num_sentences):
            for j in range(layers):
                vectors_list.append(((i,j),tensors[(i,j)]['sequence'],tensors[(i,j)]['label'],tensors[(i,j)]['dimension'],tensors[(i,j)]['vectors']))
    return dataloader(vectors_list, config)


def dataloader(vectors, config):
    sampler = SequentialSampler(vectors)
    # Definir el tamaño del lote
    batch_size = config['batch_size'] # always 12, because it is the number of attention layers, 12 layers for the same sentence
    # Crear el DataLoader sin un BatchSampler
    dataloader = DataLoader(vectors, batch_size=batch_size, sampler=sampler)
    return dataloader


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_loop(model, iterator, optimizer, criterion, config, clip = 1.0):
    #Training loop
    model.train()
    loss_sum = 0
    seed = 42
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    for i, (_,_,_,_,input) in enumerate(iterator):
        optimizer.zero_grad()
        output, _ = model(input)
        loss = criterion(output, input)
        loss.backward()
        #prevent gradients from exploding
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        #Update params
        optimizer.step()
        loss_sum += loss.item()

    epoch_train_loss = loss_sum * config['batch_size'] / len(iterator)

    return epoch_train_loss


# Extraer el vector latente fijo de cada elemento del batch
def extract_latent_vectors(model, dataloader, embedding_type):
    model.eval()
    vector_representations = {}

    with torch.no_grad():
        for (id,s,label,dim,input) in dataloader:
            latent_vectors = []
            if embedding_type == 'Attention': # Autoatenciones con LSTM
                _, (latent_representation, _) = model.encoder(input)
                latent = latent_representation.squeeze(0)
            if embedding_type == 'CLS': # CLS con AE lineal
                latent_representation = model.encoder(input)
                latent = latent_representation.squeeze(0).squeeze(1)
            tuples = list(zip(id[0].tolist(), id[1].tolist()))
            for i in range(latent.size(0)):
              latent_vectors.append(latent[i].numpy())
              vector_representations[tuples[i]] = { 'vectors' : latent[i].numpy(), 'sequences': s, 'labels': label, 'dimensions':dim}

    return vector_representations

def train_AE(instances, config):
    show_training_process(instances)
    NUM_EPOCHS = int(instances['entry_epochs'].get())
    train_loss_values = []
    history = {"train": {"loss": []}}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    input_size = int(config['layers'])  #due to 12 layers
    hidden_size = int(instances['entry_dim_reduction'].get())  # size of fixed vector
    learning_rate = float(instances['entry_lr'].get())

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    global model
    if instances['embedding_type'].get() == 'Attention': #modelo LSTM (autoatenciones)
        model = AutoencoderLSTM(input_size, hidden_size)
    if instances['embedding_type'].get() == 'CLS': #modelo lineal (CLS)
        input_size = 768
        model = LinearAutoencoder(input_size, hidden_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(NUM_EPOCHS):

        start_time = time.time()
        epoch_train_loss = train_loop(model,data_loader,optimizer,criterion,config)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        train_loss_values.append(epoch_train_loss)

        history["train"]["loss"].append(epoch_train_loss)

        #print(f'Epoch: {epoch+1:03}/{NUM_EPOCHS} | Epoch Time: {epoch_mins}m {epoch_secs}s | Train loss: {epoch_train_loss:.4f}')
        instances['text_widget'].insert('end', f'Epoch: {epoch+1:03}/{NUM_EPOCHS} | Epoch Time: {epoch_mins}m {epoch_secs}s | Train loss: {epoch_train_loss:.4f}\n')

        instances['text_widget'].update_idletasks()  # Actualizar el tamaño de la ventana
        instances['canvas_training'].configure(scrollregion=instances['canvas_training'].bbox('all'))
    show_save_embeddings(instances)

# Definir la arquitectura del autoencoder con LSTM
class AutoencoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size ):
        super(AutoencoderLSTM, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size, batch_first=True)
    def forward(self, x):
        # Codificación
        #o = [1, 12, 128] = [batch_size, len_sents, hidden_size]
        #x = [1, 12, 289] = [batch_size, len_sents, input_size]
        o, (h_n, _) = self.encoder(x)
        #h = [1, 1, 128] = [batch_size, num_layers * num_directions, hidden_size]
        # Reducción a tamaño latente
        #latent = [1, 128] = [num_layers * num_directions, hidden_size]
        latent = h_n.squeeze(0)
        # Decodificación
        #output, _ = self.decoder(latent.unsqueeze(0).repeat(1, x.size(1), 1))
        output, _ = self.decoder(o)
        return output, latent


class LinearAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(LinearAutoencoder, self).__init__()
        dims = [input_dim, 512, 256, 128, 32, 8, 2]

        layer_dims= [i for i in dims if i > latent_dim ]
        layer_dims.append(latent_dim)

        encoder_layers = []
        decoder_layers = []

        for i in range(len(layer_dims) - 1):
            encoder_layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            encoder_layers.append(nn.GELU())
            decoder_layers.insert(0, nn.Linear(layer_dims[i+1], layer_dims[i]))
            decoder_layers.insert(0, nn.GELU())

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        # Codificación
        encoded = self.encoder(x)
        # Decodificación
        decoded = self.decoder(encoded)
        return decoded, encoded



def save_embedded_reduced(instances, config, messagebox): 
    instances['label_save_pth'].config(text=f".pth guardado")
    vector_representations = extract_latent_vectors(model, data_loader,instances['embedding_type'].get())
    try:
        torch.save(vector_representations, config['root'] + instances['entry_save_reduction'].get())
        instances['btn_AE_to_clustering'].place(relx=0.8, rely=0.9, anchor=instances['tk'].CENTER)
    except Exception as e:
        messagebox.showerror("Error", "Se produjo una excepción:" + str(e))