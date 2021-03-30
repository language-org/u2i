# %% [markdown]
## Intent inference
#
# author: Steeve Laquitaine
#
# TABLE OF CONTENTS
#
# * Packages
# * Paths
# * Parameters
# * Load data
# * Filtering
#   * by query complexity
#   * by grammatical mood
#   * by syntactical similarity
# * Parsing
# * Inference
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
import os
from collections import defaultdict

import pandas as pd
import spacy

# %%
proj_path = "/Users/steeve_laquitaine/desktop/CodeHub/intent/"
os.chdir(proj_path)
from intent.src.intent.nodes import (
    features,
    inference,
    parsing,
    preprocess,
    retrieval,
    similarity,
)
from intent.src.tests import test_run

todf = pd.DataFrame
# %% [markdown]
## PATHS
# %%
cfg_path = (
    proj_path + "intent/data/02_intermediate/cfg_25_02_2021_18_16_42.xlsx"
)
sim_path = proj_path + "intent/data/02_intermediate/sim_matrix.xlsx"
# %% [markdown]
## PARAMETERS
# %%
SEED = " VB NP"  # seed for comparison
THRES_NUM_SENT = 1  # keep query with max one sentence
NUM_SENT = 1  # keep query with max one sentence
THRES_SIM_SCORE = 1  # Keep queries syntactically similar to seed
FILT_MOOD = ("ask",)  # ("state", "wish-or-excl", "ask")  # Keep statements
DIST_THRES = (
    1.8  # inference threshold for clustering, low values -> more clusters
)
# %% [markdown]
# LOAD DATA
cfg = pd.read_excel(cfg_path)
sim_matx = pd.read_excel(sim_path)
# test
test_run.test_len_similarity_matx(cfg, sim_matx)
# %% [markdown]
## FILTERING
#
### by query complexity
# %%
# cfg_cx = preprocess.filter_by_sent_count(cfg, THRES_NUM_SENT, verbose=True)
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
posting_list = retrieval.create_posting_list(tag)
sim_ranked = similarity.rank_nearest_to_seed(sim_matx, seed=SEED, verbose=True)
ranked = similarity.print_ranked_VPs(cfg_mood, posting_list, sim_ranked)
filtered = similarity.filter_by_similarity(ranked, THRES_SIM_SCORE)
# test [TODO]
test_run.test_rank_nearest_to_seed(sim_matx, seed=SEED)
test_run.test_posting_list(posting_list, sim_matx, seed=SEED)
test_run.test_get_posting_index(cfg_mood, posting_list, sim_ranked)
# %%
# %% [markdown]
## PARSING
#
# * Apply dependency parsing to each query
# * Collect intent's action (ROOT) and object (dobj)
# %%
intents = parsing.parse_intent(filtered)
# %%
## OUTPUT
#%%
# %%
# set display options
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", -1)
cfg_mood.merge(
    todf(intents, index=filtered.index), left_index=True, right_index=True
)[["text", "intent", "intendeed"]]
# %% [markdown]
## PREPROCESSING
#
# Filter out words not contained in Wordnet
# %%
filtered_corpus = preprocess.filter_words_not_in_wordnet(tuple(cfg_mood["VP"]))
# %% [markdown]
## INFERENCE
# %%
labels = inference.label_queries(filtered_corpus, DIST_THRES)
# %%
labels.sort_values(by=["label"])
