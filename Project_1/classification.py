import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import *
from sklearn.preprocessing import label_binarize
from scipy import interp
import matplotlib.pyplot as plt
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import json
from sklearn.naive_bayes import MultinomialNB
from scipy import spatial
from sklearn.multiclass import OneVsRestClassifier
import operator
import math

#klash gia statistika
class Metrics_for_Class:
    def __init__(self):
        self.rec = 0.0
        self.acc = 0.0
        self.prec = 0.0
        self.fl_sc = 0.0

def euclideanDist(x, xi):
    d = 0.0
    for i in range(len(x)-1):
        d += pow((float(x[i])-float(xi[i])),2)  #euclidean distance
    d = math.sqrt(d)
    return d

class knn_classifier():
    def __init__(self,K):
        self.K = K

    def get_params(self, deep=True):
        return {'K':self.K}

    def set_params(self, **params):
        return True

    def fit(self,array, classes):
        self.train_data = list(zip(array,classes))
        return self

    def predict(self,test_array):
        results = []
        for test in test_data:
            distances = []
            for row in self.train_data:
                distances.append(euclideanDist(row[0], test), row[1])

            sorted_data = sorted(distances)
            top = sorted_data[:self.K]
            ns = {}
            for neigh in top:
                cl = neigh[1]
                if not cl in ns:
                    ns[cl] = 1
                else:
                    ns[cl] += 1
            results.append(max(ns, key=lambda i: ns[i]))
        return results


def main():
    
    #diavazoume to csv se panda kai meta ftiaxnoume ton vectorizer + kanoume trubcated gia na meiwsoume tis diastaseis
    with open('extraStopWords.json','r') as extraStopWords:
        extraStopWords = json.load(extraStopWords)
    stopWords = ENGLISH_STOP_WORDS.union(extraStopWords)
 
    df = pd.read_csv("datasets/train_set.csv", sep="\t")
    
    count_vect = CountVectorizer(stop_words=stopWords)
    X_train_counts = count_vect.fit_transform(df.Content)
    svd = TruncatedSVD(n_components = 60)
    X_train_counts = svd.fit(X_train_counts)
        #edw dhmiourgoume to object gia na kanoume to cross validation
    kf = KFold(n_splits = 10)
    #fold = 0


    #edw exoume tous metrites pou xreiazontai
    #metrame se kathe epanalipsi to apotelesma kai sto telos kanoume total/10
    #0 einai gia svm
    #1 gia Nayve
    #2 gia Rnadom
    #3 gia KNN
    class_list = [Metrics_for_Class() for i in range(0,4)]
    
    #oi katigories
    categories = ["Technology", "Football", "Film", "Business","Politics"]
    #kratame plhroforia gia to roc plot
    folist = []
    tlist = []
    plist = []
    filist = []
    blist = []
    #edw xwrizoum
    
    for train_index, test_index in kf.split(df.Content):
        
        #pleon kanoume mono transform edw kai oxi fit dioti tha xasoume plhrofories an kanoume neo fit
        X_train_counts3 = count_vect.transform(np.array(df.Content)[train_index])
        X_train_counts2 = svd.transform(X_train_counts3)

        #to idio me to panw
        X_test_counts3 = count_vect.transform(np.array(df.Content)[test_index])
        X_test_counts2 = svd.transform(X_test_counts3)
        
        #SVM
        if sys.argv[1] == "SVM":
            #print("SVM STARTED")
            place = 0
            #parameters = {'kernel':('linear', 'rbf')}
            svr = svm.SVC(kernel = "linear")
            svr.fit(X_train_counts2, np.array(df.Category)[train_index])
            y_pred = svr.predict(X_test_counts2)
            y_true = np.array(df.Category)[test_index]
            class_list[0].rec += recall_score(y_true,y_pred,average = "macro")
            class_list[0].acc += accuracy_score(y_true,y_pred)
            class_list[0].prec += precision_score(y_true,y_pred,average = "macro")
            class_list[0].fl_sc += f1_score(y_true, y_pred,average = "macro")

            #NayveBayes
        elif sys.argv[1] == "NAYVE":
            #print("NAYVE_STARTED")
            place = 1
            clf_cv =  MultinomialNB().fit(X_train_counts3, np.array(df.Category)[train_index])
            y_pred = clf_cv.predict(X_test_counts3)
            y_true = np.array(df.Category)[test_index]
            class_list[1].rec += recall_score(y_true,y_pred,average = "macro")
            class_list[1].acc += accuracy_score(y_true,y_pred)
            class_list[1].prec += precision_score(y_true,y_pred,average = "macro")
            class_list[1].fl_sc += f1_score(y_true, y_pred,average = "macro")



        #RandomForest
        elif sys.argv[1] == "RANDOM_FOREST":
            #print("RANDOM_FOREST_STARTED")
            place = 2
            clf_rf = RandomForestClassifier(n_estimators=10).fit(X_train_counts2, np.array(df.Category)[train_index])
            y_pred = clf_rf.predict(X_test_counts2)
            y_true = np.array(df.Category)[test_index]
            class_list[2].rec += recall_score(y_true,y_pred,average = "macro")
            class_list[2].acc += accuracy_score(y_true,y_pred)
            class_list[2].prec += precision_score(y_true,y_pred,average = "macro")
            class_list[2].fl_sc += f1_score(y_true, y_pred,average = "macro")

        #KNN
        elif sys.argv[1] == "KNN":
            place = 3
            
            K = 7
            clf_kn = knn_classifier(K).fit(X_train_counts2,np.array(df.Category)[train_index])

            y_pred = clf_kn.predict(X_test_counts2,X_train_counts2,K)
            y_true = np.array(df.Category)[test_index]
            class_list[3].rec += recall_score(y_true,y_pred,average = "macro")
            class_list[3].acc += accuracy_score(y_true,y_pred)
            class_list[3].prec += precision_score(y_true,y_pred,average = "macro")
            class_list[3].fl_sc += f1_score(y_true, y_pred,average = "macro")

    #upologismos meswn orwn
    class_list[place].rec = float(class_list[place].rec) / 10
    class_list[place].acc = float(class_list[place].acc) / 10
    class_list[place].prec = float(class_list[place].prec) / 10
    class_list[place].fl_sc = float(class_list[place].fl_sc) / 10
    #class_list[place].roc_auc = float(class_list[place].roc_auc) / 10

    #print ta apotelesmata
    f = open("EvaluationMetric_" + sys.argv[1] + ".csv", "w")
    f.write("Statistic_Metrics\t")
    if sys.argv[1] == "SVM":
        f.write("SVM")
    elif sys.argv[1] == "NAYVE":
        f.write("Naive Bayes")
    elif sys.argv[1] == "RANDOM_FOREST":
        f.write("Random Forest")
    elif sys.argv[1] == "KNN":
        f.write("KNN")
    f.write("\n")
    
    #grpasimo sto csv
    f.write("Accuracy\t")
    f.write(str(class_list[place].acc) + "\n")
    f.write("Presicion\t")
    f.write(str(class_list[place].prec) + "\n")
    f.write("Recall\t")
    f.write(str(class_list[place].rec) + "\n")
    f.write("F_Measure\t")
    f.write(str(class_list[place].fl_sc) + "\n")
    f.close()

if __name__ == "__main__":
    main()