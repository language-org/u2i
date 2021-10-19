# %% [markdown]
# # Utterance exploration
#
#
# **EXECUTIVE SUMMARY **
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

import time

# data struct. utils
from collections import Counter, defaultdict

# text prep.
import nltk

# %%
# Libraries
import pandas as pd

# EDA
from ipywidgets import interact

#%%
# display

pd.set_option("display.max_colwidth", -1)

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


# %% [markdown]
# * Sentences are short, very stereotyped, the lexicon is small and extremely redundant across utterances
train_data.head(30)

# %%
