# %% [markdown]
# # Intent inference
#
# %% [markdown]
#
# author: Steeve Laquitaine
# purpose: predict intent class
# approach:
#   - preprocessing
#       - Constituency parsing
#       - Filtering
#           - complexity: keep intents w/ N sentences
#           - mood
#           - syntax similarity
#   - inference
#       - cluster and add labels
#
# TABLE OF CONTENTS
#
# * Packages
# * Parameters
# * Load data
# * Constituency parsing
# * Filtering
#   * by query complexity
#   * by grammatical mood
#   * by syntactical similarity
# * Intent parsing
# * Label inference
#
# Prerequisites
#
#   * cfg..xlsx
#   * sim_matrix.xlsx
#
# Observations:
#
#   * So far the best parameters are:
#
#       SEED            = " VB NP" <br>
#       THRES_NUM_SENT  = 1 <br>
#       NUM_SENT        = 1 <br>
#       THRES_SIM_SCORE = 1 <br>
#       FILT_MOOD       = ("ask",) <br>
#
# [TODO]:
#  - refactor and abstract pipeline
#  - link raw dataset with inference dataset with an index (primary key)
#  - Constituency parsing:
#       - Allennlp produces warning when no internet connection: "Connection error occurred while trying to fetch ETag for https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz. Will attempt to use latest cached version of resource"

# %% [markdown]
# # PACKAGES
# %%
# set project path
from inspect import TPFLAGS_IS_ABSTRACT
import os
from collections import defaultdict

proj_path = (
    "/Users/steeve_laquitaine/desktop/CodeHub/intent/"
)
os.chdir(proj_path)

from time import time

# import packages
import pandas as pd
import spacy
import yaml
from collections import Counter
import numpy as np


# import custom nodes
from intent.src.intent.nodes import (
    features,
    inference,
    parsing,
    preprocess,
    retrieval,
    similarity,
)
from intent.src.intent.pipelines.parsing import Cfg
from intent.src.intent.pipelines.similarity import Lcs
from intent.src.tests import test_run

# shorcuts
todf = pd.DataFrame

# display
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
# %% [markdown]
## PARAMETERS
# %%
SEED = " VB NP"  # seed for comparison
NUM_SENT = 1  # keep query with max one sentence
THRES_SIM_SCORE = (
    1  # Keep queries syntactically similar to seed
)
FILT_MOOD = (
    "ask",
)  # ("state", "wish-or-excl", "ask")  # Keep statements
DIST_THRES = 2  # inference threshold for clustering, low values -> more clusters
with open(
    proj_path + "intent/conf/base/parameters.yml"
) as file:
    prms = yaml.load(file)
# %% [markdown]
## LOAD DATA
# %% [markdown]
### Raw data
# %%
t0 = time()
corpus_path = (
    proj_path + "intent/data/01_raw/banking77/train.csv"
)
corpus = pd.read_csv(corpus_path)
# %% [markdown]
## PREPROCESSING
# %% [markdown]
### Constituency parsing
# %%
# [warning] this is slow
cfg = Cfg(corpus, prms).do()
# %% [markdown]
### Filtering
# %% [markdown]
#### filter complexity
# %% [markdown]
# We kept intents with N sentences
# %%
cfg_cx = preprocess.filter_n_sent_eq(
    cfg, NUM_SENT, verbose=True
)
# %% [markdown]
#### filter mood
# %%
cfg_mood = preprocess.filter_mood(cfg_cx, FILT_MOOD)
# %%
tag = parsing.chunk_cfg(cfg_mood["cfg"])
# %% [markdown]
#### filter syntax similarity
# %%
# calculate similarity
similarity_matrix = Lcs().do(cfg_mood)
test_run.test_len_similarity_matx(
    cfg_mood, similarity_matrix
)
# %%
sim_ranked = similarity.rank_nearest_to_seed(
    similarity_matrix, seed=SEED, verbose=True
)
posting_list = retrieval.create_posting_list(tag)
ranked = similarity.print_ranked_VPs(
    cfg_mood, posting_list, sim_ranked
)
filtered = similarity.filter_by_similarity(
    ranked, THRES_SIM_SCORE
)
# test [TODO]
test_run.test_rank_nearest_to_seed(
    similarity_matrix, seed=SEED
)
test_run.test_posting_list(
    posting_list, similarity_matrix, seed=SEED
)
test_run.test_get_posting_index(
    cfg_mood, posting_list, sim_ranked
)

# %%
# map back to raw intent indices
raw_ix = cfg_mood["index"]
filtered_raw_ix = raw_ix.values[filtered.index.values]
# %% [markdown]
#### Intent parsing
# %% [markdown]
# 1. Apply dependency parsing to each query
# 2. Apply NER
# 3. Retrieve (intent (ROOT), intendeed (dobj), entities (NER))
# %%
intents = parsing.parse_intent(filtered)
# %%
# show (intent, intendeed)
cfg_mood.index = cfg_mood["index"]
cfg_mood.merge(
    todf(intents, index=filtered_raw_ix),
    left_index=True,
    right_index=True,
)[["index", "text", "intent", "intendeed"]]
# %% [markdown]
## LABEL INFERENCE
#
# 1. Filter words not in Wordnet
# 2. Apply verb phrase hierarchical clustering
# %%
filtered_corpus = preprocess.filter_words(
    cfg_mood["VP"], "not_in_wordnet"
)
# %%
filtered_corpus = preprocess.drop_empty_queries(
    filtered_corpus
)
# %%
# [warning] this is very slow
tic = time()
labels = inference.label_queries(
    tuple(filtered_corpus), DIST_THRES
)
# %%
labels.index = filtered_corpus.index
print(f"{round(time() - tic, 2)} secs")
print(f"Total: {round(time() - t0, 2)} secs")
labelled = labels.sort_values(by=["label"])

# %% [markdown]
### Method 1: map cluster with True label for evaluation
# %%
# pros:
#
#   - easy to implement
#
# cons:
#
#   - wrong in some cases:
#
#       - when there are as many clusters as instances, there will be 100% accuracy
#           because each instance will take the true label as its predicted label.
#
# %%
true_labels = corpus.loc[labelled.index]["category"]
labelled["true_labels"] = true_labels
labelled
# %%
# assign the most likely label to each cluster
# %%
unique_labels = labelled["label"].unique()
predicted_labels_all = []
proba_predicted_all = []
n_labelled = len(labelled)

# for each cluster, assign its most likely true label as the predicted label
for ix, this_label in enumerate(unique_labels):

    # find indices of this cluster intents
    this_label_ix = np.where(labelled == this_label)[0]
    nb_label = len(this_label_ix)

    # find its most frequent true label
    predicted = Counter(
        labelled["true_labels"].iloc[this_label_ix].tolist()
    ).most_common()[0]

    # assign this true label as predicted label and its conditional proba
    # as proba of predicted
    predicted_labels_all += [predicted[0]] * nb_label
    proba_predicted_all += [
        predicted[1] / n_labelled
    ] * nb_label
labelled["predicted"] = predicted_labels_all
labelled["proba_predicted (ratio)"] = proba_predicted_all
labelled

# %% [markdown]
## EVALUATION
# %% [markdown]
# %% [markdown]
### Calculate performance metrics
# %%
nb_TP = sum(
    labelled["predicted"] == labelled["true_labels"]
)
accuracy = nb_TP / n_labelled
print("Task info:")
print(
    "- number of classes:",
    labelled["true_labels"].nunique(),
)
print("\nMetrics:")
print("- accuracy:", accuracy)
