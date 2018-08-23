#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import json
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

with open('extraStopWords.json','r') as extraStopWords:
	extraStopWords = json.load(extraStopWords)
stopWords = ENGLISH_STOP_WORDS.union(extraStopWords)

categories = ['Politics','Film','Football','Business','Technology']

df = pd.read_csv('./datasets/train_set.csv', sep='\t')

for category in categories:
    print("Creating word cloud for: " + category + "." )
    c_df =  df[(df['Category']==category)]
    content = ' '.join(c_df['Title'].iloc[i] + ' ' + c_df['Content'].iloc[i] for i in range(len(c_df)))
    wordcloud = WordCloud(background_color="white", stopwords=stopWords).generate(content)
    plt.imsave('WordCloud_For:_'+category+'_.png', wordcloud)
    
print("Done!")