import numpy as np 
import pandas as pd
import config_lstm as config
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.utils.np_utils import to_categorical

from gensim.models.word2vec import Word2Vec

import re


def main():

    train = pd.read_csv('train.tsv', sep="\t")

    # Phrases to lists of words
    train['Phrase'] = train['Phrase'].apply(lambda x: x.lower())
    #print (train['Phrase'])
    corpus_text = '\n'.join(train['Phrase'])
    phrases = corpus_text.split('\n')
    phrases = [line.lower().split(' ') for line in phrases]

    #print (phrases)

    # Create Word2Vec
    word2vec = Word2Vec(sentences=phrases,
                        size=config.w2v_vector_size, 
                        window=config.w2v_window_size,
                        min_count=config.w2v_min_count) 

    pretrained_weights = word2vec.wv.syn0
    vocab_size, embed_dim = pretrained_weights.shape

    # Get corresponding index/word
    def word2idx(word):
        #print (word + ": " + str(word2vec.wv.vocab[word].index))
        return word2vec.wv.vocab[word].index
    def idx2word(idx):
        return word2vec.wv.index2word[idx]

    # Set up training inputs and outputs    
    max_phrase_len = max([len(phrase) for phrase in phrases])
    train_x = np.zeros([len(phrases), max_phrase_len], dtype=np.int32)
    train_y = np.zeros([len(phrases)], dtype=np.int32)

    for i, phrase in enumerate(phrases):
        for t, word in enumerate(phrase):
            train_x[i, t] = word2idx(word)
        train_y[i] = train['Sentiment'][i]
    train_y = to_categorical(train_y)

    print('train_x shape:', train_x.shape)
    print('train_y shape:', train_y.shape)

    # Define LSTM model
    def create_model(num_layers=1, num_units=128, dropout = 0.1,
                     recurrent_dropout = 0.1, optimizer = 'adam'):
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=embed_dim,
                            weights=[pretrained_weights]))
#        model.add(SpatialDropout1D(0.4))
        for i in range(num_layers):
            # Last layer we don't need to return sequences
            if(i==num_layers-1):
                # Halve the number of units for every LSTM layer added
                model.add(LSTM(int(num_units/(2**i)), dropout=dropout, 
                               recurrent_dropout=recurrent_dropout))
            else:
                model.add(LSTM(int(num_units/(2**i)), dropout=dropout, 
                               recurrent_dropout=recurrent_dropout,
                               return_sequences=True))
        model.add(Dense(5,activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                      metrics=['accuracy'])
        print(model.summary())
        return model

    # sklearn wrapper so we can use Grid Search
    model = KerasClassifier(build_fn=create_model)
    param_grid = dict(num_units=config.num_units, dropout=config.dropout, 
                      recurrent_dropout=config.recurrent_dropout,
                      optimizer=config.optimizer,
                      validation_split=config.validation_split,
                      batch_size=config.batch_size, epochs=config.epochs,
                      num_layers=config.num_layers)
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(train_x, train_y)

    # Summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

if __name__ == '__main__':
    main()
