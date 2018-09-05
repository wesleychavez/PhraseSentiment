import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    train = pd.read_csv('train.tsv', sep="\t")
    print (train.dtypes)    
    print (train.shape)    
    print (train.describe())

    # Mean number of words per phrase
    print (np.mean(train['Phrase'].apply(lambda x: len(x.split()))))

    # Save a histogram of the sentiments
    fig, ax = plt.subplots()
    train.hist('Sentiment', ax=ax, bins=[-0.5,0.5,1.5,2.5,3.5,4.5])
    fig.savefig('SentHist.png')



if __name__ == '__main__':
    main()
