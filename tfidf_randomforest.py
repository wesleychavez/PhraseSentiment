# tfidf_logreg.py - Performs logistic regression on the phrase/sentiments with k-fold
#                   cross-validation, and prints one-vs-one and one-vs-all
#                   validation accuracy.
#
# Usage: python -B tfidf_logreg.py
#
# Wesley Chavez 08-27-18

import config_rf
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def main():
    train = pd.read_csv('train.tsv', sep="\t")
    test = pd.read_csv('test.tsv', sep="\t")
    tfidf = TfidfVectorizer()

    x_train = tfidf.fit_transform(train['Phrase'])
    y_train = train['Sentiment']

    print ("x_train: " + str(x_train.shape))

    # oob_score: Use hold-out examples for validation
    rf = RandomForestClassifier(n_estimators=config_rf.n_est, oob_score=True)
    rf.fit(x_train, y_train)
    print("Accuracy: " + str(rf.oob_score_))

if __name__ == '__main__':
    main()
