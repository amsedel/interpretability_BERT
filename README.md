# Install project

pip install virtualenv      #Install virtual environment

virtualenv venv     #Create virtual environment on Unix/Linux/macOS
python -m venv venv     #Create virtual environment on Windows

source venv/bin/activate        # To activate the virtual environment on Unix/Linux systems
.\venv\Scripts\activate         # To activate the virtual environment on Windows

pip install -r requirements.txt         #Install project dependencies 

deactivate      #Deactivate the virtual environment



# linguistic-interpretability-of-BERT-with-unsupervised

visual tool in python to perform the analysis of the BERT transformer model through unsupervised learning

For interpretability analysis, it is necessary to train the model to solve semantic similarity using the CLS token (yielding better results compared to proposed aggregation methods and LSTM networks).

First, attentions are obtained from the trained model, in this case for the STS-Benchmark and SICK-R test sets (but feel free to experiment with other datasets).

The interpretability analysis accepts the .pth (PyTorch) format. However, it is recommended to save in the .pth format only if the number of examples is small when saving attention scores. For saving attentions from the SICK-R dataset (4906 examples), the free memory in Colab is not sufficient. Therefore, for analysis, only the file of vectors reduced to 2 dimensions was saved, which is later loaded into the GUI (option Load Reduced Vectors). For the STS-B dataset (1379 examples), it was possible to save and load both the file of trimmed attentions (for subsequent dimensionality reduction using the Autoencoder option in the GUI) and the representations in 2 dimensions.

To save the generated trimmed attentions from training to solve semantic similarity in the STS-B set, you can refer to the notebook:
https://github.com/amsedel/semantic_similarity_BERT/blob/main/STS_Benchmark/get_attentions_and_2d_vectors/FineTunningSTS_B_CLS_token_GENERATE_cls_attentions_outputs_for_AUTOENCODER.ipynb

To save the generated trimmed attentions from training to solve semantic similarity in the SICK-R set, you can refer to the notebook:
https://github.com/amsedel/semantic_similarity_BERT/blob/main/SICK/get_attentions_and_2d_vectors/FineTunningSICK_GENERATE_attentions_outputs_and_autoencoder.ipynb

Once in the notebooks, train the model and execute the `save_attentions` function to extract the trimmed attentions and generate a dictionary with the form:

```python
all_attentions[(example_number, layer, head)]
```

Note: Remember that for the BERT base model, there are 12 layers and 12 heads per layer.

Subsequently, the `all_attentions` dictionary can be saved in .pth format for a subsequent dimensionality reduction process either in the visual GUI tool or through the notebooks:

For STS-Benchmark:
https://github.com/amsedel/semantic_similarity_BERT/blob/main/STS_Benchmark/get_attentions_and_2d_vectors/Autoencoder_RNN_forAttentions_STS_B.ipynb

For SICK-R:
https://github.com/amsedel/semantic_similarity_BERT/blob/main/SICK/get_attentions_and_2d_vectors/FineTunningSICK_GENERATE_attentions_outputs_and_autoencoder.ipynb

Once you have the vectors reduced to 2 dimensions, you can perform attention analysis through the GUI option "Load Reduced Vectors."

In the repository, the files reduced to two dimensions for attention analysis are included:

`reduced_vectors_att_sick_2.pth`
`reduced_vectors_att_sts_2.pth`




It is also possible to perform an analysis using PCA using a reduced representation of the CLS token or the same attentions reduced to higher dimensions than 2 dimensions by varying the size of the latent vector of the autoencoders.

To obtain a reduced version of the CLS token, you can execute the following notebook to extract the CLS vectors (768 dimensions for BERT base) using the `save_CLS_outputs` function:
https://github.com/amsedel/semantic_similarity_BERT/blob/main/STS_Benchmark/PCA_CLS_vectors/FineTunningSTS_B_CLS_token_GENERATE_cls_outputs_for_AUTOENCODER.ipynb

Subsequently, you can use a linear autoencoder to reduce to a smaller dimension (or, if space permits, use the original 768-dimensional vectors) using the following notebook:
https://github.com/amsedel/semantic_similarity_BERT/blob/main/STS_Benchmark/PCA_CLS_vectors/Autoencoder_Linear_forCLS_STS_B.ipynb

Note: Due to the size of the SICK set, the free memory in Colab is not sufficient to save the 768-dimensional vectors.

In the repository, there are example files for PCA analysis of the STS-benchmark dataset:

reduced_vectors_32.pth
pca_CLS_64.pth