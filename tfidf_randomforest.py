# tfidf_randomforest.py - Random forest on the tf-idf vectorized phrases/sentiments
#                         with out-of-bag examples (hold-out set) used as validation.
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
# Usage: python -B tfidf_randomforest.py
#
# Wesley Chavez 09-05-18

import config_rf as config
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

def main():
    train = pd.read_csv('train.tsv', sep="\t")
    test = pd.read_csv('test.tsv', sep="\t")
    tfidf = TfidfVectorizer()

    x_train = tfidf.fit_transform(train['Phrase'])
    y_train = train['Sentiment']

    print ("x_train: " + str(x_train.shape))

    # oob_score: Use hold-out examples for validation
    rf = RandomForestClassifier(n_estimators=config.n_est, oob_score=True)
    rf.fit(x_train, y_train)
    print("Accuracy: " + str(rf.oob_score_))

if __name__ == '__main__':
    main()
