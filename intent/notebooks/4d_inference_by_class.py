# %% [markdown]
#
# # Intent inference
#
# author: Steeve Laquitaine
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

# %% [markdown]
## PACKAGES
# %%
# set project path
import os
from collections import defaultdict

proj_path = "/Users/steeve_laquitaine/desktop/CodeHub/intent/"
os.chdir(proj_path)

from time import time

# import packages
import pandas as pd
import spacy
import yaml

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
pd.set_option("display.max_colwidth", -1)
# %% [markdown]
## PARAMETERS
# %%
SEED = " VB NP"  # seed for comparison
NUM_SENT = 1  # keep query with max one sentence
THRES_SIM_SCORE = 1  # Keep queries syntactically similar to seed
FILT_MOOD = ("ask",)  # ("state", "wish-or-excl", "ask")  # Keep statements
DIST_THRES = (
    5  # inference threshold for clustering, low values -> more clusters
)
with open(proj_path + "intent/conf/base/parameters.yml") as file:
    prms = yaml.load(file)
# %% [markdown]
## LOAD DATA
# %%
t0 = time()
corpus_path = proj_path + "intent/data/01_raw/banking77/train.csv"
corpus = pd.read_csv(corpus_path)
# %% [markdown]
## CONSTITUENCY PARSING
# %%
cfg = Cfg(corpus, prms).do()
# %% [markdown]
## FILTERING
#
### by complexity
# %%
cfg_cx = preprocess.filter_n_sent_eq(cfg, NUM_SENT, verbose=True)
# %% [markdown]
### by grammatical mood
# %%
cfg_mood = preprocess.filter_in_only_mood(cfg_cx, FILT_MOOD)
# %%
tag = parsing.from_cfg_to_constituents(cfg_mood["cfg"])
# %% [markdown]
### by syntactical similarity
# %%
# calculate similarity
similarity_matrix = Lcs().do()
test_run.test_len_similarity_matx(cfg, similarity_matrix)
sim_ranked = similarity.rank_nearest_to_seed(
    similarity_matrix, seed=SEED, verbose=True
)
posting_list = retrieval.create_posting_list(tag)
ranked = similarity.print_ranked_VPs(cfg_mood, posting_list, sim_ranked)
filtered = similarity.filter_by_similarity(ranked, THRES_SIM_SCORE)

# test [TODO]
test_run.test_rank_nearest_to_seed(similarity_matrix, seed=SEED)
test_run.test_posting_list(posting_list, similarity_matrix, seed=SEED)
test_run.test_get_posting_index(cfg_mood, posting_list, sim_ranked)
# %% [markdown]
## INTENT PARSING
#
# 1. Apply dependency parsing to each query
# 2. Apply NER
# 3. Retrieve (intent (ROOT), intendeed (dobj), entities (NER))
# %%
intents = parsing.parse_intent(filtered)
# %%
# show (intent, intendeed)
cfg_mood.merge(
    todf(intents, index=filtered.index), left_index=True, right_index=True
)[["text", "intent", "intendeed"]]
# %% [markdown]
## LABEL INFERENCE
#
# 1. Filter words not in Wordnet
# 2. Apply verb phrase hierarchical clustering
# %%
tic = time()
filtered_corpus = preprocess.filter_words_not_in_wordnet(tuple(cfg_mood["VP"]))
filtered_corpus = preprocess.filter_empty_queries(filtered_corpus)
# filtered_corpus = preprocess.filter_ambiguous_postag(filtered_corpus)
labels = inference.label_queries(filtered_corpus, DIST_THRES)
print(f"{round(time() - tic, 2)} secs")
print(f"Total: {round(time() - t0, 2)} secs")
labels.sort_values(by=["label"])

