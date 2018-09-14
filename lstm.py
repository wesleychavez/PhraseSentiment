import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import itertools
import config_lstm as config

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.utils.np_utils import to_categorical
from gensim.models.word2vec import Word2Vec

def main():

    train = pd.read_csv('train.tsv', sep="\t")
    test = pd.read_csv('test.tsv', sep="\t")

    # Phrases to lists of words
    train['Phrase'] = train['Phrase'].apply(lambda x: x.lower())
    test['Phrase'] = test['Phrase'].apply(lambda x: x.lower())
    corpus_text_train = '\n'.join(train['Phrase'])
    corpus_text_test = '\n'.join(test['Phrase'])
    phrases_train = corpus_text_train.split('\n')
    phrases_test = corpus_text_test.split('\n')
    phrases_train = [line.lower().split(' ') for line in phrases_train]
    phrases_test = [line.lower().split(' ') for line in phrases_test]
    phrases = phrases_train + phrases_test


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
    train_x = np.zeros([len(phrases_train), max_phrase_len], dtype=np.int32)
    train_y = np.zeros([len(phrases_train)], dtype=np.int32)

    for i, phrase in enumerate(phrases_train):
        for t, word in enumerate(phrase):
            train_x[i, t] = word2idx(word)
        train_y[i] = train['Sentiment'][i]
    train_y = to_categorical(train_y)

    print('train_x shape:', train_x.shape)
    print('train_y shape:', train_y.shape)

    # Define LSTM model
    def create_model(num_layers=1, num_units=128, dropout=0.1,
                     recurrent_dropout=0.1, optimizer='adam'):
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=embed_dim,
                            weights=[pretrained_weights],
                            input_length=max_phrase_len))
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

    param_grid = [config.num_layers, config.num_units, config.dropout, 
                  config.recurrent_dropout, config.optimizer, config.batch_size,
                  config.epochs, config.validation_split]
    combos = list(itertools.product(*param_grid))
    max_validation_accuracies = []
    # Grid Search, plotting cross-validation scores by epoch
    for i in range(len(combos)):

        model = create_model(num_layers=combos[i][0], num_units=combos[i][1],
                             dropout=combos[i][2], recurrent_dropout=combos[i][3],
                             optimizer=combos[i][4])

        history = model.fit(x=train_x, y=train_y, batch_size=combos[i][5],
                  epochs=combos[i][6], validation_split=combos[i][7])
        max_validation_accuracies.append(max(history.history['val_acc']))	
        fig, ax = plt.subplots(1,2)
        ax[0].plot(history.history['acc'])
        ax[0].plot(history.history['val_acc'])
        ax[0].set_title('Model Accuracy')
        ax[0].set_ylabel('Accuracy')
        ax[0].set_xlabel('Epoch')
        ax[0].legend(['Train', 'Test'], loc='upper left')

        ax[1].plot(history.history['loss'])
        ax[1].plot(history.history['val_loss'])
        ax[1].set_title('Model Loss')
        ax[1].set_ylabel('Loss')
        ax[1].set_xlabel('Epoch')
        ax[1].legend(['Train', 'Test'], loc='upper left')

        out_name = 'lstm_out' + str(combos[i])
        out_name = out_name.replace(" ", "")
        out_name = out_name.replace("'", "")
        out_name = out_name.replace("(", "_")
        out_name = out_name.replace(")", "_")
        out_name = out_name.replace(",", "_")
        fig.savefig(out_name + '_accuracyandloss.png')

    print('Model with hightest validation accuracy:')
    print(combos[max_validation_accuracies.index(max(max_validation_accuracies))])
    print('Max validation accuracy:')
    print(max(max_validation_accuracies))
    for i in range(len(max_validation_accuracies)):
        print(combos[i])
        print(max_validation_accuracies[i])

if __name__ == '__main__':
    main()
