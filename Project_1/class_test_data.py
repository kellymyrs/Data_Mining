import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import json
from sklearn.naive_bayes import MultinomialNB
import numpy as np

#tha xrhsimopoihsw ton liner dioti htane o kaluteros
def main():
    #ta idia opws kai sto classification
    df = pd.read_csv("datasets/train_set.csv", sep="\t")
    with open('extraStopWords.json','r') as extraStopWords:
        extraStopWords = json.load(extraStopWords)
    stop_words = ENGLISH_STOP_WORDS.union(extraStopWords)
    count_vect = CountVectorizer(stop_words=stop_words)
    X_train_counts = count_vect.fit_transform(df.Content)

    #pairnoume ta test
    df2 = pd.read_csv("datasets/test_set.csv", sep="\t")
    X_test_counts = count_vect.transform(df2.Content)

    #SVM Linear dioti htane o kaluteros
    clf_cv =  MultinomialNB().fit(X_train_counts, np.array(df.Category))
    y_pred = clf_cv.predict(X_test_counts)

    f = open("testSet_categories.csv", "w")
    f.write("ID\tPredicted_Category\n")
    i = 0
    for pred in y_pred:
        f.write(str(df2.Id[i]) + "\t" + pred + "\n")
        i += 1
    f.close()
    
if __name__ == "__main__":
    main()
    print("Done")
