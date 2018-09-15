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
    
    # Save a histogram of the sentiments of phrases containing "!"
    exclamation = train[train['Phrase'].str.contains('!')]

    fig, ax = plt.subplots()
    train_heights, train_bins = np.histogram(train['Sentiment'], bins=[-0.5,0.5,1.5,2.5,3.5,4.5])
    exclamation_heights, exclamation_bins = np.histogram(exclamation['Sentiment'], bins=[-0.5,0.5,1.5,2.5,3.5,4.5])

    # Normalize by maximum
    train_heights = train_heights/np.max(train_heights)
    exclamation_heights = exclamation_heights/np.max(exclamation_heights)

    # Plot both histograms
    width = (train_bins[1] - train_bins[0])/3
    train_plt = ax.bar(train_bins[:-1], train_heights, width=width, facecolor='cornflowerblue')
    excl_plt = ax.bar(exclamation_bins[:-1]+width, exclamation_heights, width=width, facecolor='seagreen')
    plt.title('Distribution of Sentiments in dataset, full data vs phrases containing "!"')
    plt.legend([train_plt,excl_plt],['Full', '!'], fontsize='x-small', loc='upper right')
    fig.savefig('SentHist_all_vs_!.png')

if __name__ == '__main__':
    main()
