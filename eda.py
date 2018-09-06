import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    train = pd.read_csv('train.tsv', sep="\t")

    print ("\n")
    print (train.dtypes)    
    print ("\n")
    print (train.shape)    
    print ("\n")
    print (train.describe())

    # Mean number of words per phrase
    print ("\nMin, Mean, Max words per phrase:")
    print (np.min(train['Phrase'].apply(lambda x: len(x.split()))))
    print (np.mean(train['Phrase'].apply(lambda x: len(x.split()))))
    print (np.max(train['Phrase'].apply(lambda x: len(x.split()))))

    # Save a histogram of the sentiments
    fig, ax = plt.subplots()
    train.hist('Sentiment', ax=ax, bins=[-0.5,0.5,1.5,2.5,3.5,4.5])
    fig.savefig('SentHist.png')



if __name__ == '__main__':
    main()
