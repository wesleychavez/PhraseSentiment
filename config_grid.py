# config_lstm.py - Model hyperparameters

# Word2Vec 
w2v_vector_size = 128
w2v_window_size = 5
w2v_min_count = 1

# LSTM
layer_type = ['lstm','gru','bilstm','bigru']
num_layers = [1,2]
num_units = [512]
dropout = [0.1]
recurrent_dropout = [0.1]
spatial_dropout = [0.1]
optimizer = ['adam']
batch_size = [32]
epochs = 3
k = 3
