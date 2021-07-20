# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Utterance clustering
#
#
# **EXECUTIVE SUMMARY **
# * **corpus**:
#     * Banking77 dataset
# * **analysis**
#   * data preparation
#   * k-means clustering
#   * Workload (< 1 hour)
#
# * **Impact**
#     * dataset automatic labelling --> reduce labor intensive annotation
#
# * [TODO]
#     * convert queries to triples (intendee,intent,intendeed)
#     * plot
#
# **TABLE OF CONTENT**
# * **Set path**
# * **Set parameters**
# * **Load corpus**
# * **Normalize headers**
# * **Describe**
# * **Cluster**
#     * **Cluster queries**
#     * **Explore clusters**

# %%
# Libraries
import os
import pandas as pd

# ml.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    plot_confusion_matrix,
    classification_report,
)
from sklearn.cluster import KMeans

# text prep.
import nltk
import re
import numpy as np

nltk.download(
    "punkt"
)  # 13 MB zip containing pretrained punkt sentence tokenizer (Kiss and Strunk, 2006)
import time

# data struct. utils
from collections import defaultdict

# EDA
from ipywidgets import interact
from collections import Counter

# %% [markdown]
# ### Set paths

# %%
proj_path = "/Users/steeve_laquitaine/desktop/CodeHub/intent/intent/"
train_data_path = proj_path + "data/01_raw/banking77/train.csv"
test_data_path = proj_path + "data/01_raw/banking77/test.csv"

# %% [markdown]
# ### Set parameters

# %%
params = defaultdict()

# tf-idf
params["tfidf"] = defaultdict()
params["tfidf"]["MIN_DF"] = 10
params["tfidf"]["MAX_DF"] = 0.8

# stop words
params["stop_words"] = nltk.corpus.stopwords.words("english")

# kmeans
params["kmeans"] = defaultdict()
params["kmeans"]["NUM_CLUSTERS"] = 6
params["kmeans"]["max_iter"] = 1000
params["kmeans"]["n_init"] = 50
params["kmeans"]["random_state"] = 42

# %% [markdown]
# ### Load corpus

# %%
# train and test
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)


# %%
# show
train_data.head(5)


# %%
# preview
test_data.head(5)

# %% [markdown]
# ### Normalize headers

# %%
def standardize_col_names(data: pd.DataFrame):
    return data.rename(columns={"text": "text", "category": "intent"})


# %%
train_data = standardize_col_names(train_data)
test_data = standardize_col_names(test_data)

# %% [markdown]
# ### Describe

# %%
print("\nValue count:\n")
print(train_data.count())
print("\nUnique values:\n")
print(train_data.nunique())


# %%
train_data.head(1)

# %% [markdown]
# ## Cluster Queries

# %%
# prepare queries
stop_words = nltk.corpus.stopwords.words("english")


def normalize_document(doc: list):
    """
    Normalize document

    parameters:
    ---------
    doc:list

    return
    ------
    doc: array
    """
    # lower case and drop special characters\whitespaces
    doc = re.sub(r"[^a-zA-Z0-9\s]", "", doc, re.I | re.A)
    doc = doc.lower()
    doc = doc.strip()
    tokens = nltk.word_tokenize(doc)  # tokenize
    filtered_tokens = [
        token for token in tokens if token not in stop_words
    ]  # drop stop words
    doc = " ".join(filtered_tokens)  # re-create doc from filtered tokens
    return doc


tic = time.time()  # time
normalize_corpus = np.vectorize(normalize_document)  # vectorize doc
norm_corpus = normalize_corpus(list(train_data["text"]))  # normalize doc
len(norm_corpus)
print(f"(normalize_document) took: {round(time.time()-tic,2)} secs")
print("\nPreview:")
norm_corpus


# %%
## Vectorize queries as B-O-W
tic = time.time()
cv = CountVectorizer(
    ngram_range=(1, 2),
    min_df=params["tfidf"]["MIN_DF"],
    max_df=params["tfidf"]["MAX_DF"],
    stop_words=params["stop_words"],
)
cv_matrix = cv.fit_transform(norm_corpus)
print(
    f"(vectorization:tf-idf) shape:{cv_matrix.shape}, took {round(time.time()-tic,2)} secs"
)


# %%
## Cluster queries w/ K-Means
tic = time.time()
km = KMeans(
    n_clusters=params["kmeans"]["NUM_CLUSTERS"],
    max_iter=params["kmeans"]["max_iter"],
    n_init=params["kmeans"]["n_init"],
    random_state=params["kmeans"]["random_state"],
).fit(cv_matrix)
print(f"(clustering:kmeans) model:{km}, took {round(time.time()-tic,2)} secs")


# %%
# count queries per cluster
Counter(km.labels_)


# %%
kmeans_labelled_train_data = train_data.copy(deep=True)
kmeans_labelled_train_data["kmeans_label"] = km.labels_
kmeans_labelled_train_data.head()

# %% [markdown]
# ## Explore Clusters

# %%
# interactive
@interact(LABEL=np.unique(km.labels_), text_ix=(0, len(kmeans_labelled_train_data)))
def show_requests(LABEL, text_ix):
    VIEW_WINDOW = 10
    return kmeans_labelled_train_data[
        kmeans_labelled_train_data.kmeans_label.eq(LABEL)
    ].text.iloc[text_ix : text_ix + VIEW_WINDOW]


# %% [markdown]
#
# %% [markdown]
# # References
# %% [markdown]
# (1) https://www.nltk.org/_modules/nltk/ccg/chart.html
# (2) https://github.com/dipanjanS/text-analytics-with-python/blob/master/New-Second-Edition/Ch07 - Text Similarity and Clustering/Ch07c - Document Clustering.ipynb
# (3) https://www.martechvibe.com/insights/staff-articles/how-intent-analysis-can-help-programmatic-advertising/#:~:text=Intent analysis is a step,user's intention behind the message.

