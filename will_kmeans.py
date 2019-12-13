#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 10:28:19 2019

@author: christopherton
"""

# Trying to Cluster DataFest words

from __future__ import print_function
import psycopg2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.cluster import KMeans 
import pyperclip

pd.__version__

# Create Dataframe Object
color = pd.Series(['Red', 'White', 'Rose'])
type = pd.Series(['Still', 'Sparkling', 'Fortified'])

pd.DataFrame({ 'color': color, 'wine_type': type })

# Datafest wine Marks
PGHOST="datafest201912.library.ucdavis.edu"
PGDATABASE="postgres"
PGPORT="49152"
PGUSER="anon"
PGPASSWORD="anon"

conn_string = ("host={} port={} dbname={} user={} password={}") \
  .format(PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD)

conn=psycopg2.connect(conn_string)

# edit SQL string here
sql_command = "SELECT * FROM {}.{};".format("datafest", "mark")
marks = pd.read_sql(sql_command, conn)
marks.describe()

# All pages 
sql_command = "select page_id, p.page_ark from page p"
page = pd.read_sql(sql_command, conn)
page
row_idx = page.index[page['page_ark'] == "d7q36x-009"].tolist()
row_contents = page.iloc[row_idx]

# Save link to clipboard so you can paste into browser and view catalouge
page_link = 'https://datafest201912.library.ucdavis.edu'+row_contents['page_id'].values[0]
pyperclip.copy(page_link)

# Individual Words
sql_command = "SELECT * FROM {} WHERE page_ark='{}';".format("rtesseract_words","d7q36x-009")
words = pd.read_sql(sql_command, conn)
words

# Importing the Dataset
X = words.iloc[:, [2,3]].values

# Plotting the words based on bottom left point of word
plt.scatter(words['left'], words['top'])
plt.show()

print(words.shape)

### height of words > mean, conf >80
height = words["bottom"] - words["top"] 
height = pd.DataFrame({"height" : height})
words  = words.join(height)
words= words.sort_values(by='height', ascending=False)
words = words[words["confidence"]>80]


print(words.describe())

# Only use if you want to filter the words by soem height threshold 

#==============================================================================
# large_words = words[words["height"] > 50 ]
# print(large_words.describe())
# print(large_words.shape)
#==============================================================================



# Using the elbow method to find number of clusters 
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Applying k-means to the dataset
kmeans = KMeans(n_clusters = 4,  init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)


# Visualising the clusters 
def all_plot():
    
    plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], c='red')
    plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], c='blue')
    plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], c='green')
    plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], c='cyan')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
    plt.title('Cluster of Words (by bottom-left-most point)')
    plt.xlabel('Left')
    plt.ylabel('Bottom')
    plt.show()
    
all_plot()

def sep_plot():
    plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], c='red')
    plt.title('Cluster 1')
    plt.show()
    plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], c='blue')
    plt.title('Cluster 2')
    plt.show()
    plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], c='green')
    plt.title('Cluster 3')
    plt.show()
    plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], c='cyan')
    plt.title('Cluster 4')
    plt.show()
    #==============================================================================
    # plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1], c='yellow')
    # plt.title('Cluster 5')
    # plt.show()
    # plt.scatter(X[y_kmeans==5, 0], X[y_kmeans==5, 1], c='black')
    # plt.title('Cluster 6')
    # plt.show()
    #==============================================================================

# Find the left and bottom vals of the top-most word in each cluster
data_0 = {'left': X[y_kmeans==0, 0], 'bottom': X[y_kmeans==0, 1]}
cluster_0_df = pd.DataFrame(data_0)
data_1 = {'left': X[y_kmeans==1, 0], 'bottom': X[y_kmeans==1, 1]}
cluster_1_df = pd.DataFrame(data_1)

lowest_word = min(data_0['bottom'])
lowest_word

# Extract the index of this word from the 
words_to_return = words.index[words['top'] == lowest_word]
words_to_return
words_content = words.iloc[words_to_return]