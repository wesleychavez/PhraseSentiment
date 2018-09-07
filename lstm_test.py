import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.utils.np_utils import to_categorical

from gensim.models.word2vec import Word2Vec

import re


def main():

    train = pd.read_csv('simple.tsv', sep="\t")

    # Phrases to lists of words
    train['Phrase'] = train['Phrase'].apply(lambda x: x.lower())
    print (train['Phrase'])
    corpus_text = '\n'.join(train['Phrase'])
    phrases = corpus_text.split('\n')
    phrases = [line.lower().split(' ') for line in phrases]

    print (phrases)

    # Word2vec params
    vector_size = 5
    window_size = 2
        
    # Create Word2Vec
    word2vec = Word2Vec(sentences=phrases,
                        size=vector_size, 
                        window=window_size,
                        min_count=1) 

    pretrained_weights = word2vec.wv.syn0
    vocab_size, embed_dim = pretrained_weights.shape


    def word2idx(word):
        print (word + ": " + str(word2vec.wv.vocab[word].index))
        return word2vec.wv.vocab[word].index
    def idx2word(idx):
        return word2vec.wv.index2word[idx]

    max_phrase_length = 4 
    train_x = np.zeros([len(phrases), max_phrase_length], dtype=np.int32)
    train_y = np.zeros([len(phrases)], dtype=np.int32)

    for i, phrase in enumerate(phrases):
        for t, word in enumerate(phrase):
            train_x[i, t] = word2idx(word)
        train_y[i] = train['Sentiment'][i]
    train_y = to_categorical(train_y)

    print('train_x shape:', train_x.shape)
    print('train_y shape:', train_y.shape)
    print('train_x:', train_x)
    print('train_y:', train_y)


    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embed_dim,
                                     weights=[pretrained_weights]))
#    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(128, dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(4,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',
                                                metrics = ['accuracy'])
#    print(model.summary())


    # fit the model
    model.fit(train_x, train_y,
              batch_size=32,
              epochs=10)
    # evaluate the model

#    loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
#    print('Accuracy: %f' % (accuracy*100))

if __name__ == '__main__':
    main()
