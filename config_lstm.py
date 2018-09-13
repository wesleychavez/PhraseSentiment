# config_lstm.py - Model hyperparameters

# Word2Vec 
w2v_vector_size = 128
w2v_window_size = 5
w2v_min_count = 1

# LSTM
num_layers = [1,2]
num_units = [256]
dropout = [0.1,0.4]
recurrent_dropout = [0.1,0.4]
optimizer = ['adam']
validation_split = [0.33]
batch_size = [32]
epochs = [2]
