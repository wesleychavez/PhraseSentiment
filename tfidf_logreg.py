# tfidf_logreg.py - Performs logistic regression on the tf-idf vectorized
#                   phrase/sentiments with k-fold cross-validation, and 
#                   prints one-vs-one and one-vs-all validation accuracy.
#
# Movie review phrases are in English, and are overlapping.  Sentiments of these
# phrases are judged by Mechanical Turkers and are integers from 0-4.
# Data from https://www.kaggle.com/artgor/movie-review-sentiment-analysis-eda-and-models/data
#
# 13 example phrases and corresponding sentiments:
#
# have a hard time sitting through this one	0
# have	2
# a hard time sitting through this one	1
# a hard time	1
# hard time	1
# hard	2
# time	2
# sitting through this one	1
# sitting	2
# through this one	2
# through	2
# this one	2
# one	2
#
#
#
# Usage: python -B tfidf_logreg.py
#
# Wesley Chavez 08-27-18

import config_lr as config
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import cross_val_score

def main():
    train = pd.read_csv('train.tsv', sep="\t")
    test = pd.read_csv('test.tsv', sep="\t")
    tfidf = TfidfVectorizer()

    x_train = tfidf.fit_transform(train['Phrase'])
    print ("x_train: " + str(x_train.shape))
    y = train['Sentiment']

    logreg = LogisticRegression()
    ovr = OneVsRestClassifier(logreg)
    ovo = OneVsOneClassifier(logreg)

    scores_ovr = cross_val_score(ovr, x_train, y, scoring='accuracy', n_jobs=-1, cv=config.k)    
    scores_ovo = cross_val_score(ovo, x_train, y, scoring='accuracy', n_jobs=-1, cv=config.k)        

    print('Cross-validation mean validation accuracy for OVR {0:.2f}%, std {1:.2f}.'.format(np.mean(scores_ovr) * 100, np.std(scores_ovr) * 100))
    print('Cross-validation mean validation accuracy for OVO {0:.2f}%, std {1:.2f}.'.format(np.mean(scores_ovo) * 100, np.std(scores_ovo) * 100))

if __name__ == '__main__':
    main()
